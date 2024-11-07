import torch


def train_epoch_one_step(
    dataloader, model, loss_fn, optimizer, device
) -> tuple[float, dict]:
    model.train()
    num_batches = len(dataloader)
    acc_train_loss = 0

    for X, y in dataloader:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        acc_train_loss += loss.item()

        loss.backward()
        optimizer.step()
    return acc_train_loss / num_batches, {}


def train_epoch_unrolled(
    dataloader_with_unrolling, model, loss_fn, optimizer, device, **kwargs
) -> tuple[float, dict]:
    horizon_size = dataloader_with_unrolling.dataset.__dict__.get(
        "unrolling_horizon", 1
    )
    if horizon_size == 1:
        return train_epoch_one_step(
            dataloader_with_unrolling, model, loss_fn, optimizer, device
        )
    model.train()
    num_batches = len(dataloader_with_unrolling)
    acc_train_loss = 0
    loss_hist = []

    detach_for: set[int] = kwargs.get("detach_for", set())

    for input_val, next_steps_true in dataloader_with_unrolling:
        optimizer.zero_grad()
        input_val, next_steps_true = input_val.to(device), next_steps_true.to(device)
        next_steps_pred = [input_val]
        for unr_step_id in range(horizon_size):
            next_steps_pred.append(
                model(next_steps_pred[-1]).detach()
                if unr_step_id in detach_for
                else model(next_steps_pred[-1])
            )
        next_steps_pred_torch = torch.concat(next_steps_pred[1:], dim=1)

        loss = loss_fn(next_steps_pred_torch, next_steps_true)
        acc_train_loss += loss.item()
        loss_hist.append(loss.item())
        loss.backward()

        optimizer.step()
    return acc_train_loss / num_batches / horizon_size, {"loss_hist": loss_hist}


def test_epoch_one_step(dataloader, model, loss_fn, device) -> float:
    model.eval()
    num_batches = len(dataloader)
    acc_test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            acc_test_loss += loss_fn(pred, y).item()
    return acc_test_loss / num_batches


def test_epoch_unrolled(dataloader_with_unrolling, model, loss_fn, device) -> float:
    horizon_size = dataloader_with_unrolling.dataset.__dict__.get(
        "unrolling_horizon", 1
    )
    if horizon_size == 1:
        return test_epoch_one_step(dataloader_with_unrolling, model, loss_fn, device)
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
            for _ in range(horizon_size):
                next_steps_pred.append(model(next_steps_pred[-1]))
            next_steps_pred_torch = torch.concat(next_steps_pred[1:], dim=1)
            acc_test_loss += loss_fn(next_steps_pred_torch, next_steps_true).item()
    return acc_test_loss / num_batches / horizon_size


def train_epoch(
    dataloader, model, loss_fn, optimizer, device, unrolling_horizon, **kwargs
) -> tuple[float, dict]:
    if unrolling_horizon == 1:
        return train_epoch_one_step(dataloader, model, loss_fn, optimizer, device)
    return train_epoch_unrolled(dataloader, model, loss_fn, optimizer, device, **kwargs)


def test_epoch(dataloader, model, loss_fn, device, unrolling_horizon) -> float:
    if unrolling_horizon == 1:
        return test_epoch_one_step(dataloader, model, loss_fn, device)
    return test_epoch_unrolled(dataloader, model, loss_fn, device)
