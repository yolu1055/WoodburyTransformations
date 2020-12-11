import torch
import os


def save_model(model, optim, scheduler, dir, iteration):
    path = os.path.join(dir, "checkpoint_{}.pth.tar".format(iteration))
    state = {}
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None

    torch.save(state, path)



def load_state(path, cuda):
    if cuda:
        print ("load to gpu")
        state = torch.load(path)

    else:
        print ("load to cpu")
        state = torch.load(path, map_location=lambda storage, loc: storage)

    return state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
