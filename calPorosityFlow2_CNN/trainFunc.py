from tqdm.auto import tqdm
import torch


def rescaleCTToN1and1(ct):
    CTG = 2232.875
    W = 0
    AIR = -1000

    rescale = (ct - AIR) * (1 / (CTG - AIR))
    rescale = 2 * rescale - 1
    return rescale


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn=None,
               device: torch.device = 'cpu'):
    ''' Return train_loss and train_acc'''
    train_loss, train_acc = 0, 0
    model.to(device)

    for batch, (X, y) in enumerate(tqdm(data_loader)):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        if accuracy_fn is not None:
            train_acc += accuracy_fn(
                y_true=y,
                y_pred=y_pred.argmax(dim=1))  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    if accuracy_fn is not None:
        train_acc /= len(data_loader)
        print(
            f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    else:
        print(f"Train loss: {train_loss:.5f}")
    return train_loss, train_acc


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn=None,
              device: torch.device = 'cpu'):
    ''' Return test_loss and test_acc'''
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            if accuracy_fn is not None:
                test_acc += accuracy_fn(
                    y_true=y,
                    y_pred=test_pred.argmax(
                        dim=1)  # Go from logits -> pred labels
                )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        if accuracy_fn is not None:
            test_acc /= len(data_loader)
            print(
                f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n"
            )
        else:
            print(f"Test loss: {test_loss:.5f}\n")
        return test_loss, test_acc