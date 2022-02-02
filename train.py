import argparse
from util import *
from model.ODCRN import *
from tqdm import tqdm
from load_data import *
from tensorboardX import SummaryWriter
from collections import defaultdict
import math
import copy,time

parser = argparse.ArgumentParser()
#parameter of dataset
parser.add_argument('--train_prop',type=float,default=0.8,help='proportion of training set')
parser.add_argument('--val_prop',type=float,default=0.1,help='proportion of validation set')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--height',type=int,default=16,help='input flow image height')
parser.add_argument('--width',type=int,default=16,help='input flow image width')
parser.add_argument('--external_dim',type=int,default=28,help='external factor dimension')
parser.add_argument('--num_grids',type=int,default=256,help='')
parser.add_argument('--hist_len',type=int,default=8,help='')

#parameter of training
parser.add_argument('--epochs',type=int,default=200,help='training epochs')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--seed',type=int,default=99,help='running seed')
parser.add_argument('--save_folder',type=str,default='./result',help='result dir')
parser.add_argument('--device',type=str,default='cuda:0',help='cuda device')
parser.add_argument('--max_grad_norm',type=int,default=5,help='max gradient norm for gradient clip')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
#parameter of model

parser.add_argument('--input_dim', default=256, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--output_dim', default=256, type=int)
parser.add_argument('--gnn_type', default='gat', type=str)
parser.add_argument('--use_svd', default=True, type=bool)
parser.add_argument('--svd_dim', default=10, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--embed_dim', default=10, type=int)
parser.add_argument('--predict_steps', default=1, type=int)
parser.add_argument('--num_nodes', default=256, type=int)
parser.add_argument('--num_rows', default=16, type=int)
parser.add_argument('--num_cols', default=16, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--input_len', default=8, type=int)
parser.add_argument('--rnn_units', default=128, type=int)
parser.add_argument('--cuda', default=True, type=bool)

parser.add_argument('--static_feat_hiddens',type=list, default=[10,32,32],help='node2vec embeddings')
parser.add_argument('--dynamic_feat_hiddens',type=list, default=[[1,32,32],[32,32,32]],help='in out flow embeddings')
parser.add_argument('--context_fusion_type',type=str, default='concat',help='context info fusion type')
parser.add_argument('--use_context_info',type=bool, default=True,help='whether use context infomation')
parser.add_argument('--max_diffusion_step',type=int, default=2,help='max_diffusion_step')


args = parser.parse_args()


def train(model,
          dataloaders,
          optimizer,
          epochs,
          folder,
          loss_func,
          graph,
          node_embedding,
          early_stop_steps = 10,
          device = 'cpu',
          max_grad_norm = None):

    #1. save path
    save_path = os.path.join(folder,'model', 'best_model.pkl')
    tensorboard_folder = os.path.join(folder,'tensorboard')

    if os.path.exists(save_path):
        print('path exists')
        save_dict = torch.load(save_path)
        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1

        #move the load parameter tensor to cuda, for optimizer.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        print('path not exists')
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch  = 0


    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)

    writer = SummaryWriter(tensorboard_folder)
    since = time.perf_counter()

    phases = ['train', 'validate']
    model = model.to(device)

    node_embed = torch.from_numpy(node_embedding)
    graph = torch.tensor(graph, dtype=torch.float)

    #2. train model
    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_loss, running_metrics = defaultdict(float), dict()
            print('epoch=',epoch)

            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                tqdm_loader = tqdm(enumerate(dataloaders[phase]))

                steps, pred, targets = 0, list(), list()
                for step, data in tqdm_loader:
                    #input data
                    x, y, crowd_flow, _ = data['od_input'],data['od_label'],data['flow_input'],data['flow_label']
                    targets.append(y.numpy())

                    with torch.no_grad():
                        x = x.to(device)
                        y = y.to(device)
                        graph = graph.to(device)
                        node_embed = node_embed.to(device).float()
                        crowd_flow = crowd_flow.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        out = model(x, graph, crowd_flow, node_embed)
                        loss = loss_func(out, y)

                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()

                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        pred.append(out.cpu().numpy())

                    running_loss[phase] += loss * len(x)
                    steps += len(x)
                    torch.cuda.empty_cache()

                running_metrics[phase] = evaluate(np.concatenate(pred),np.concatenate(targets),dataloaders['scaler'])

                #for select model
                if phase == 'validate':
                    if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')

                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            for phase in phases:
                for key,val in running_metrics[phase].items():
                    writer.add_scalars(f'{phase}', {f'{key}': val}, global_step=epoch)
                    writer.add_scalars('Loss', {
                        f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                                       global_step=epoch)
                    print(f'epoch:{epoch},{phase} loss:{running_loss[phase]},{key}:{val}')
    except:
        writer.close()
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

    writer.close()


def test_model(folder,
               model,
               dataloaders,
               graph,
               node_embedding,
               device):

    save_path = os.path.join(folder, 'model', 'best_model.pkl')
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model = model.to(device)

    print(model)
    # for name,param in model.named_parameters():
    #     print('param name:{}, param shape:{}'.format(name,param.shape))

    # model.eval()
    steps, pred, targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(dataloaders['test']))

    graph = torch.tensor(graph, dtype=torch.float)
    node_embed = torch.from_numpy(node_embedding)

    for step, data in tqdm_loader:
        x, y, crow_flow, _ = data['od_input'], data['od_label'], data['flow_input'], data['flow_label']
        targets.append(y.numpy())

        with torch.no_grad():
            x = x.to(device)
            graph = graph.to(device)
            node_embed = node_embed.to(device).float()
            crow_flow = crow_flow.to(device)

            out = model(x, graph, crow_flow, node_embed)
            pred.append(out.cpu().numpy())

    pred, targets = np.concatenate(pred,axis=0), np.concatenate(targets,axis=0)
    scores = evaluate(pred, targets, dataloaders['scaler'])

    print('test results:')
    print(json.dumps(scores,cls=MyEncoder, indent=4))

    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f,cls=MyEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), pred=pred, targets=targets)

def main():
    # 1.set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    #2. load data
    dataloader_s = get_dataloader(hist_len=args.hist_len, num_grids=args.num_grids, batch_size=args.batch_size,\
                                   train_prop=args.train_prop,val_prop=args.val_prop)

    _, graph = convert_geo_adj(args.num_rows, args.num_cols)

    node_embedding = load_node_embedding()

    #3. construct model
    model = ODCRN(
        args.num_nodes,
        args.hidden_dim,
        args.num_layers,
        args.static_feat_hiddens,
        args.dynamic_feat_hiddens,
        args.max_diffusion_step
    )
    print(model)

    #4. optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    loss_func = nn.MSELoss()

    #5. train
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train(model = model,
        dataloaders = dataloader_s,
        optimizer = optimizer,
        epochs = args.epochs,
        folder = args.save_folder,
        loss_func = loss_func,
        graph = graph,
        node_embedding = node_embedding,
        early_stop_steps=10,
        device=device,
        max_grad_norm=args.max_grad_norm)


    #6. test
    test_model(folder = args.save_folder,
               model = model,
               dataloaders = dataloader_s,
               graph = graph,
               node_embedding=node_embedding,
               device = device)

if __name__ == '__main__':
    main()