import wandb
from train import eval_ensemble, eval

def test_model_performance(
    model1, model2, test_lab_dl
):
    """
    tests a segmentation model.

    The dataloader 'test_lab_dl' should contain only labeled images
    and is only used for the supervised loss.

    """
    # print statistics
    test_loss1, test_acc1, test_iou1, test_dice1 = eval(model1, test_lab_dl)
    test_loss2, test_acc2, test_iou2, test_dice2 = eval(model2, test_lab_dl)
    test_loss, test_acc, test_iou, test_dice = eval_ensemble(model1, model2, test_lab_dl)
    wandb.log(
        {
            "test_loss1": test_loss1,
            "test_acc1": test_acc1,
            "test_iou1": test_iou1,
            "test_dice1": test_dice1,
            "test_loss2": test_loss2,
            "test_acc2": test_acc2,
            "test_iou2": test_iou2,
            "test_dice2": test_dice2,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_iou": test_iou,
            "test_dice": test_dice,
        }
    )

    return  test_acc1, test_acc2, test_iou1, test_iou2, test_dice1, test_dice2, test_acc, test_iou, test_dice

