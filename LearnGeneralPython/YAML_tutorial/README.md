# How to use YAML to control config (in machine learning)?
Let's first write a simple script, include data loading, model defining, training, etc. (this has been done in *main_script.py*)

However, there are certain hyperparameters that can be changed. Without a proper config control, we need to change the *main_script.py* file each time we made any changes. This is quite inefficient in a way, can we do better?

This is why we need to use YAML to help manage the config. 

A *.yaml* file is quite simple to understand, you can treat it as a dictionary. Let's see *config/my_config.yaml*, the **data_name: data.csv** is a quite simple **key:value** pair, isn't it? Yes, this is how yaml file works. 

In most of cases, you will also need argparse to communicate with CLI, so you can pass different config via command lines. Of course, you should also log your results for training, like *tensorboard, wandb*, but this is not the focus in this tutorial.

Now let's jump into the *main.py* to see how we construct the code.

First of all, you need a function to load yaml file into dictionary
~~~
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
~~~
The ouput of this function is a dictionary that can be used for passing all the configrations to the *main()*.

Then based on the yaml file, we need to replace all the keys inside the *yaml* to **config[KEYS]**. For example, we can replace **data_name** to **config["data_name"]** after defining *config* dict:
~~~
    config = load_config(os.path.join(config_path, args['config']))
~~~

Finally (optional), if you want to use terminal to communicate your code with different configs, you can use 
~~~
# before the def main():
def get_argparse():
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="the config file name")
    args = vars(ap.parse_args())
    return args
~~~
and 
~~~
# within the def main():
args = get_argparse()
config_path = os.path.join(CONFIG_PATH, 'config')
config = load_config(os.path.join(config_path, args['config']))
~~~

After this, you can use command line
~~~
$ python main.py -c CONFIG_NAME.yaml
~~~

