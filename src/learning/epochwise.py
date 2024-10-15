import torch


def train_loop(dataloader, model, loss_fn, optimizer, device) -> float:
    model.train()
    num_batches = len(dataloader)
    acc_train_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        acc_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return acc_train_loss / num_batches


def train_loop_with_unrolling(
    dataloader_with_unrolling, model, loss_fn, optimizer, device
) -> float:
    model.train()
    num_batches = len(dataloader_with_unrolling)
    acc_train_loss = 0

    for input_val, next_steps_true in dataloader_with_unrolling:
        input_val, next_steps_true = input_val.to(device), next_steps_true.to(device)
        next_steps_pred = [input_val]
        for _ in range(dataloader_with_unrolling.dataset.get_horizon_size()):
            next_steps_pred.append(model(next_steps_pred[-1]))
        next_steps_pred_torch = torch.concat(next_steps_pred[1:], dim=1)
        loss = loss_fn(next_steps_pred_torch, next_steps_true)
        acc_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return acc_train_loss / num_batches


def test_loop(dataloader, model, loss_fn, device) -> float:
    model.eval()
    num_batches = len(dataloader)
    acc_test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            acc_test_loss += loss_fn(pred, y).item()
    return acc_test_loss / num_batches


def test_loop_with_unrolling(
    dataloader_with_unrolling, model, loss_fn, device
) -> float:
    model.eval()
    num_batches = len(dataloader_with_unrolling)
    acc_test_loss = 0

    with torch.no_grad():
        for input_val, next_steps_true in dataloader_with_unrolling:
            input_val, next_steps_true = (
                input_val.to(device),
                next_steps_true.to(device),
            )
            next_steps_pred = [input_val]
            for _ in range(dataloader_with_unrolling.dataset.get_horizon_size()):
                next_steps_pred.append(model(next_steps_pred[-1]))
            next_steps_pred_torch = torch.concat(next_steps_pred[1:], dim=1)
            acc_test_loss += loss_fn(next_steps_pred_torch, next_steps_true).item()
    return acc_test_loss / num_batches
