from parse_args import parse_args, get_log_name
import torch
from torch_geometric.utils import degree
import numpy as np
from nets import SAGENet, GATNet

from logger import LightLogging
from sklearn.metrics import accuracy_score, f1_score
from utils import load_dataset, build_loss_op, build_sampler
import pandas as pd
from time import time
from metanet import Filter, Buffer
from utils import filter_, calc_avg_loss
# from metric_and_loss import MetaLoss


log_path = './logs'
summary_path = './summary'
torch.manual_seed(2020)


def train_sample(norm_loss, loss_op):
    model.train()
    model.set_aggr('add')
    meta_sampler.eval()
    total_loss = total_examples = 0
    for data in loader:


        optimizer.zero_grad()

############# Meta Sampler section ################
        with torch.no_grad():
            x = buffer.get_x_rank(data.n_id).to(device)
            data = data.to(device)
            meta_prob = meta_sampler(x, data.edge_index)
            data.x *= meta_prob

############# End ###################################
        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)

        else:
            out = model(data.x, data.edge_index)

############## Meta Sampler Section ###################
        loss = loss_op(out, data)
        avg_loss = calc_avg_loss(loss)
        prob = out.softmax(dim=-1)
        #print(loss.size(), 'ls')
        #buffer.update_prob_each_class(data.n_id[data.train_mask].cpu(), prob[data.train_mask].detach().cpu())
        buffer.update_avg_train_loss(data.n_id[data.train_mask].cpu(), avg_loss[data.train_mask].detach().cpu())
##########################################################
        loss = loss[data.train_mask].mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    #print(data.n_id)
    return total_loss / total_examples


def train_full(loss_op):
    model.train()
    model.set_aggr('mean')

    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))

    loss = loss_op(out[data.train_mask], data.y.to(device)[data.train_mask]).mean()

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_full():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    out = out.log_softmax(dim=-1)
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    return accs


@torch.no_grad()
def eval_full_multi():
    model.eval()
    model.set_aggr('mean')
    out = model(data.x.to(device), data.edge_index.to(device))
    out = (out > 0).float().cpu().numpy()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        score = f1_score(data.y[mask], out[mask], average='micro')
        accs.append(score)
    return accs


def eval_sample(norm_loss):
    model.eval()
    model.set_aggr('add')
    meta_sampler.train()
    res_df_list = []
    for data in loader:
###########################Meta################
        meta_optimizer.zero_grad()
        x = buffer.get_x_rank(data.n_id).to(device)

        data = data.to(device)
        meta_prob = meta_sampler(x, data.edge_index.to(device))
        data.x = meta_prob * data.x
###############################################
        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        prob = out.softmax(dim=-1)
        loss = loss_op(out, data)
        mask = data.train_mask + data.val_mask

        buffer.update_best_valid_loss(data.n_id[mask].cpu(), loss[mask].detach().cpu())
        buffer.update_prob_each_class(data.n_id[mask].cpu(), prob[mask].detach().cpu())

        loss = loss.mean()
        loss.backward()

        meta_optimizer.step()

        out = out.log_softmax(dim=-1)
        pred = out.argmax(dim=-1)

        res_batch = pd.DataFrame()
        res_batch['nid'] = data.indices.cpu().numpy()
        res_batch['pred'] = pred.cpu().numpy()
        res_df_list.append(res_batch)
    res_df_duplicate = pd.concat(res_df_list)
    start_time = time()
    tmp = res_df_duplicate.groupby(['nid', 'pred']).size().unstack().fillna(0)
    res_df = pd.DataFrame()
    res_df['nid'] = tmp.index
    res_df['pred'] = tmp.values.argmax(axis=1)
    # res_df = res_df.groupby('nid')['pred'].apply(lambda x: np.argmax(np.bincount(x))).reset_index()  # 10s

    res_df.columns = ['nid', 'pred']
    res_df = res_df.merge(node_df, on=['nid'], how='left')

    accs = res_df.groupby(['mask']).apply(lambda x: accuracy_score(x['y'], x['pred'])).reset_index()
    accs.columns = ['mask', 'acc']
    accs = accs.sort_values(by=['mask'], ascending=True)
    accs = accs['acc'].values

    return accs


def eval_sample_multi(norm_loss):
    model.eval()
    model.set_aggr('add')

    res_df_list = []
    accs = [[], [], []]
    for data in loader:
        meta_optimizer.zero_grad()
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        x = buffer.get_x_rank(data.n_id).to(device)
        data = data.to(device)
        meta_prob = meta_sampler(x, data.edge_index)
        data.x = meta_prob * data.x


        if norm_loss == 1:
            out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        else:
            out = model(data.x, data.edge_index)

        prob = out.softmax(dim=-1)
        loss = loss_op(out, data)
        avg_loss = calc_avg_loss(loss)
        total_loss = torch.mean(avg_loss)
        total_loss.backward()
        mask = train_mask + val_mask

        buffer.update_best_valid_loss(data.n_id[mask].cpu(), avg_loss[mask].detach().cpu())
        buffer.update_prob_each_class(data.n_id[mask].cpu(), prob[mask].detach().cpu())


        meta_optimizer.step()

        # prob[prob >= 0.5] = 1
        # prob[prob < 0.5] = 0
        #
        # accs[0].append(f1_score(data.y[train_mask].cpu().numpy(), prob[train_mask].cpu().numpy(), average='micro'))
        # accs[1].append(f1_score(data.y[val_mask].cpu().numpy(), prob[val_mask].cpu().numpy(), average='micro'))
        # accs[2].append(f1_score(data.y[test_mask].cpu().numpy(), prob[test_mask].cpu().numpy(), average='micro'))
        # return [np.mean(i) for i in accs]
        res_batch = (out > 0).float().cpu().numpy()
        res_batch = pd.DataFrame(res_batch)
        res_batch['nid'] = data.indices.cpu().numpy()
        res_df_list.append(res_batch)

    res_df_duplicate = pd.concat(res_df_list)
    length = res_df_duplicate.groupby(['nid']).size().values
    tmp = res_df_duplicate.groupby(['nid']).sum()
    nid = tmp.index
    masks = []
    for l_nid in [train_nid, val_nid, test_nid]:
        masks.append([np.in1d(nid, l_nid.nonzero()), np.intersect1d(l_nid.nonzero(), nid)])
    prob = tmp.values
    res_matrix = []
    for i in range(prob.shape[1]):
        a = prob[:, i] / length
        a[a >= 0.5] = 1
        a[a < 0.5] = 0
        res_matrix.append(a)
    res_matrix = np.array(res_matrix).T
    accs = []
    for mask_r, mask_l in masks:
        accs.append(f1_score(label_matrix[mask_l], res_matrix[mask_r], average='micro'))
    return accs



def func(x):
    if x in train_nid:
        return 0
    elif x in val_nid:
        return 1
    elif x in test_nid:
        return 2
    else:
        return -1


if __name__ == '__main__':

    args = parse_args(config_path='./default_hparams.yml')
    log_name = get_log_name(args, prefix='test')
    if args.save_log == 1:
        logger = LightLogging(log_path=log_path, log_name=log_name)
    else:
        logger = LightLogging(log_name=log_name)

    logger.info('Model setting: {}'.format(args))

    dataset = load_dataset(args.dataset)
    logger.info('Dataset: {}'.format(args.dataset))

    if args.dataset in ['flickr', 'reddit']:
        is_multi = False
    else:
        is_multi = True

    data = dataset[0]
    row, col = data.edge_index
    data.edge_attr = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
    data.indices = torch.arange(0, data.num_nodes)
    data.n_id = data.indices
    data.y = data.y.long()
    #print(data.y.size(), 'ys', data.num_nodes, dataset.num_classes)



    # todo add it into dataset or rewrite it in easy way
    if not is_multi:
        node_df = pd.DataFrame()
        node_df['nid'] = range(data.num_nodes)
        node_df['y'] = data.y.cpu().numpy()
        node_df['mask'] = -1
        train_nid = data.indices[data.train_mask].numpy()
        test_nid = data.indices[data.test_mask].numpy()
        val_nid = data.indices[data.val_mask].numpy()
        node_df['mask'] = node_df['nid'].apply(lambda x: func(x))
    else:
        train_nid = data.indices[data.train_mask].numpy()
        test_nid = data.indices[data.test_mask].numpy()
        val_nid = data.indices[data.val_mask].numpy()
        label_matrix = data.y.numpy()

    loader, msg = build_sampler(args, data, dataset.processed_dir)
    logger.info(msg)

    ### inductive settings######
    # train_data = filter(data, data.train_mask.nonzero().squeeze())
    # full_loader, msg2 = build_sampler(args, data, dataset.precessed_dir)
    # logger.info(msg2)
    #########################################

    if args.use_gpu == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    Net = {'sage': SAGENet, 'gat': GATNet}.get(args.gcn_type)
    logger.info('GCN type: {}'.format(args.gcn_type))
    model = Net(in_channels=dataset.num_node_features,
                hidden_channels=256,
                out_channels=dataset.num_classes, drop_out=args.drop_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_op = build_loss_op(args)


    #Meta Model
    meta_sampler = Filter(in_channels=dataset.num_classes+2,
                          hidden_channels=16,
                          out_channels=1, drop_out=args.meta_drop_out).to(device)
    meta_optimizer = torch.optim.Adam(meta_sampler.parameters(), lr=args.meta_learning_rate)

    #Buffer
    buffer = Buffer(num_nodes=data.num_nodes,
                          num_classes=dataset.num_classes,
                    y=data.y)
    # todo replace by tensorboard
    summary_accs_train = []
    summary_accs_test = []

    for epoch in range(1, args.epochs + 1):
        if args.train_sample == 1:
            loss = train_sample(norm_loss=args.loss_norm, loss_op=loss_op)
        else:
            loss = train_full(loss_op=loss_op)
        if args.eval_sample == 1:
            if is_multi:
                accs = eval_sample_multi(norm_loss=args.loss_norm)
            else:
                accs = eval_sample(norm_loss=args.loss_norm)
        else:
            if is_multi:
                accs = eval_full_multi()
            else:
                accs = eval_full()
        if epoch % args.log_interval == 0:
            logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train-acc: {accs[0]:.4f}, '
                        f'Val-acc: {accs[1]:.4f}, Test-acc: {accs[2]:.4f}')

        summary_accs_train.append(accs[0])

        summary_accs_test.append(accs[2])

    summary_accs_train = np.array(summary_accs_train)
    summary_accs_test = np.array(summary_accs_test)

    logger.info('Experiment Results:')
    logger.info('Experiment setting: {}'.format(log_name))
    logger.info('Best acc: {}, epoch: {}'.format(summary_accs_test.max(), summary_accs_test.argmax()))

    summary_path = summary_path + '/' + log_name + '.npz'
    np.savez(summary_path, train_acc=summary_accs_train, test_acc=summary_accs_test)
    logger.info('Save summary to file')
    logger.info('Save logs to file')
