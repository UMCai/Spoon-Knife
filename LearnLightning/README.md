# LearnLightning

## learning track
1. [Lightning in 15 minutes](https://lightning.ai/docs/pytorch/stable/starter/introduction.html#)
2. Basic skills
    * Basic L1 - L2
        * L.Trainer(model, trainloader, validloader) control num of epoch, and can overwrite max batch size
        * L.LightningModule is super useful
    * Basic L3 fine tune
    * Basic L4 use CLI and argparse
    * Basic L5 for debugging and checking  
        * fast_dev_run
        * limit_train_batches, limit_val_batches
        * num_sanity_val_steps
        * _self.example_input_array = torch.tensor((a,b,c,d))_ in \_\_init\_\_
        * _self.log_ and _self.log_dict_
        * the whole logs in lightning can be checked by tenosrboard
    * Basic L6 checkpoint in Lightning is fully compatible with pytorch
        * torch.load(CKPT_PATH) or .load_from_checkpoint(CKPT_PATH)
3. Tutorial
    * tutorial 1 -- useless, pass
    * tutorial 2 -- useful:
        * how to customize activation function
        * how to visualize gradient distribution across layers
    * tutorial 3 -- useful:
        * visualization of optimization curve 
        * understand the relation between initialization and performance
    * tutorial 4 -- useful:
        * visualization of augmented 2D images
        * best tutorial for checking famous CNN models with torch lightning
        * clever idea to placehold models in lightning framework

## Reference

https://lightning.ai/docs/pytorch/stable/tutorials.html

also check 
https://github.com/UMCai/Monai_tutorial/blob/main/modules/TorchIO_MONAI_PyTorch_Lightning.ipynb