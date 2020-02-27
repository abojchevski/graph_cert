import numpy as np
import torch
import torch.nn.functional as F
from graph_cert.certify import k_squared_parallel


def train(model, attr, ppr, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50, adver_config=None):
    """Train a model using either standard or adversarial training.

    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    ppr: torch.Tensor [n, n]
        Dense Personalized PageRank matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    adver_config : dict
        Dictionary encoding the parameters for adversarial training.

    Returns
    -------
    trace_val: list
        A list of values of the validation loss during training.
    """

    idx_observed = np.concatenate((idx_train, idx_val))
    arng = torch.arange(len(idx_observed))
    # use the ground-truth labels as reference labels during training (in contrast to predicted labels)
    reference_labels = labels[idx_observed]

    trace_val = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf
    best_acc = -np.inf

    for it in range(max_epochs):
        logits, diffused_logits = model(attr=attr, ppr=ppr)

        if adver_config is not None:
            k_squared_pageranks = k_squared_parallel(adj=adver_config['adj_matrix'], alpha=adver_config['alpha'],
                                                     fragile=adver_config['fragile'],
                                                     local_budget=adver_config['local_budget'],
                                                     logits=logits.cpu().detach().numpy(),
                                                     nodes=idx_observed)

            # worst-case/adversarial logits
            adv_logits = get_adv_logits(logits=logits, reference_labels=reference_labels, arng=arng,
                                        k_squared_pageranks=k_squared_pageranks)

            if adver_config['loss_type'] == 'rce':
                loss_train = rce_loss(adv_logits=adv_logits[:len(idx_train)], labels=labels[idx_train])
                loss_val = rce_loss(adv_logits=adv_logits[len(idx_train):], labels=labels[idx_val])
            elif adver_config['loss_type'] == 'cem':
                margin = adver_config['margin']
                loss_train = cem_loss(diffused_logits=diffused_logits[idx_train], adv_logits=adv_logits[:len(idx_train)],
                                      labels=labels[idx_train], margin=margin)
                loss_val = cem_loss(diffused_logits=diffused_logits[idx_val], adv_logits=adv_logits[len(idx_train):],
                                    labels=labels[idx_val], margin=margin)
            else:
                raise ValueError('loss type not recognized.')

            # compute the fraction of train/val nodes which are certifiably robust
            worst_margins = adv_logits.clone()
            worst_margins[arng, reference_labels] = float('inf')
            p_robust = (worst_margins.min(1).values > 0).sum().item() / len(reference_labels)
        else:
            # standard cross-entropy
            p_robust = -1
            loss_train = ce_loss(diffused_logits=diffused_logits[idx_train], labels=labels[idx_train])
            loss_val = ce_loss(diffused_logits=diffused_logits[idx_val], labels=labels[idx_val])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        trace_val.append(loss_val.item())

        acc_train = accuracy(labels, diffused_logits, idx_train)
        acc_val = accuracy(labels, diffused_logits, idx_val)

        if loss_val < best_loss or acc_val > best_acc:
            best_loss = loss_val
            best_acc = acc_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            print(f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} '
                  f' acc_train: {acc_train:.5f}, acc_val: {acc_val:.5f} p_robust {p_robust}'
                  )

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_val


def get_adv_logits(logits, reference_labels, arng, k_squared_pageranks):
    """Compute the worst-case/adversarial logits.

    Parameters
    ----------
    logits: torch.Tensor [?, nc]
        The logits before diffusion.
    reference_labels: torch.Tensor [?, nc]
        The ground-truth or predicted class labels.
    arng: torch.Tensor [?]
        A range of numbers from 0 to ?
    k_squared_pageranks: array-like [nc, nc, ?, n]
        PageRank vectors of the perturbed graphs for all nc x nc pairs of classes.
    Returns
    -------
    adv_logits: torch.Tensor [?, nc]
        The worst-case logits for each class for a batch of nodes.
    """
    logit_diff = (logits[:, reference_labels][:, None, :] - logits[:, :, None]).transpose(0, 2)
    worst_pprs = torch.from_numpy(k_squared_pageranks).to('cuda')[reference_labels, :, arng]
    adv_logits = (logit_diff * worst_pprs).sum(2)

    return adv_logits


def ce_loss(diffused_logits, labels):
    """Compute the standard cross-entropy loss.

    Parameters
    ----------
    diffused_logits: torch.Tensor, [?, nc]
        Logits diffused by Personalized PageRank.
    labels: torch.Tensor [?]
        The ground-truth labels.

    Returns
    -------
    loss: torch.Tensor
        Standard cross-entropy loss.
    """
    return F.cross_entropy(diffused_logits, labels)


def rce_loss(adv_logits, labels):
    """Compute the robust cross-entropy loss.

    Parameters
    ----------
    adv_logits: torch.Tensor, [?, nc]
        The worst-case logits for each class for a batch of nodes.
    labels: torch.Tensor [?]
        The ground-truth labels.

    Returns
    -------
    loss: torch.Tensor
        Robust cross-entropy loss.
    """
    return F.cross_entropy(-adv_logits, labels)


def cem_loss(diffused_logits, adv_logits, labels, margin):
    """ Compute the robust hinge loss.

    Parameters
    ----------
    diffused_logits: torch.Tensor, [?, nc]
        Logits diffused by Personalized PageRank.
    adv_logits: torch.Tensor, [?, nc]
        The worst-case logits for each class for a batch of nodes.
    labels: torch.Tensor [?]
        The ground-truth labels.
    margin : int
        Margin.

    Returns
    -------
    loss: torch.Tensor
        Robust hinge loss.
    """
    hinge_loss_per_instance = torch.max(margin - adv_logits, torch.zeros_like(adv_logits)).sum(1) - margin
    loss_train = F.cross_entropy(diffused_logits, labels, reduce=False) + hinge_loss_per_instance
    return loss_train.mean()


def accuracy(labels, logits, idx):
    """ Compute the accuracy for a set of nodes.

    Parameters
    ----------
    labels: torch.Tensor [n]
        The ground-truth labels for all nodes.
    logits: torch.Tensor, [n, nc]
        Logits for all nodes.
    idx: array-like [?]
        The indices of the nodes for which to compute the accuracy .

    Returns
    -------
    accuracy: float
        The accuracy.
    """
    return (labels[idx] == logits[idx].argmax(1)).sum().item() / len(idx)
