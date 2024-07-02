# Spoon - Knife
**Auther: Shizhe Cai**
**Maastricht University**


This whole repo will be reform


This repo was originally used as a tutorial material on how to use fork and git clone. However, although many geeks may simply delete this repe due to its uselessness, I decided to make it become my trick bag on my way learning programming (python).

## Learning Plans for Repo
Folder name | ref
---------     | ---------
CommandLines | [link](https://pynative.com/python-glob/#:~:text=Python%20glob.,UNIX%20shell%2Dstyle%20wildcards)
LearnDeepReg | [link](https://github.com/DeepRegNet/DeepReg)
LearnDocker  | [link](https://docs.docker.com/)
LearnGeneralPython| --
LearnHF | [link](https://huggingface.co/docs)
LearnLightning| [link](https://lightning.ai/docs/pytorch/stable/tutorials.html)
LearnOpen3D| [link](https://www.open3d.org/docs/release/index.html) 
LearnRAG   | --
LearnTorchIO | [link](https://github.com/fepegar/torchio)



## Learning Tutorial algorithm
```
T: a set of tutorials to learn, including text or video
I: most important information
I(t): a subset of I, most information from tutorial t
P: potential project to contribute 

For t in T:
    1. briefly check all content of t (~5 mins), decide:
    if t is usefull for P:
        2. find the most important part I(t) to learn, decide:
        if I is hard to understand:
            3. practice N times until understand I(t)
        else:
            continue
        4. take notes on I(t) as reminder
    else:
        continue next t

For I(t) in I:
    if I(t) directly related to P:
        1. analyze functionality of I(t)
        2. merge into P
    3. think potential use-case of I(t)

```


## Research Innovation Algorithm (deep learning related)
```
K: the code for research
RQ: research question 
E: experiments that validate the research 

0. create a new repo and make a very good README file 
1. design a RQ, and its related E
2. build up bedrock code K for baseline RQ and E
3. define the innovation section IS from K
4. change only the IS with same input and output
5. run multiple experiments to valid the RQ

```



# Learning track

## LearnLightning

Learning goal:
1. learn the official tutorials from Lightning docs,
2. learn the difference between pytorch training and lgihtning, whaty is pros and cons
3. learn how to use lightning existed recipes to build up training framework for tasks like classification, segmentation, object detection ...
4. learn how to customize the lightning existed framework elements, like loss function, optimizer, models, and metrics (with other packages)
5. learn how to build a small project that can segment 3D medical images by using config to control different styles (__medical image segmentation__)
    * use basic augmentation methods, affine, elasitc
    * use at least three models with fine-tuning ability
    * use metrics that can valid the results easily (use wandb)
    * identify the switchable part of this project, dataset, loss, augmentation, models, optimizer and optimization methods
    * build a inference function as a finish touch

Folder: _LearnLightning_

Details are included in ./LearnLightning/README.md


