###########################
# This tutorial is about how to use argparse to communacate with terminal


def get_argparse():
	# import the necessary packages
	import argparse
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	# by using add_argument() to add the necessary arguments
	ap.add_argument("-n", "--name", required=True, help="name of the user")
	# sometimes, you can use default = XXX to set a default value
	ap.add_argument("-p", "--place", required = False, default = 'USA', help = 'the place of the user')
	# wrap up all the added arguments into a dictionary structure
	args = vars(ap.parse_args())
	return args

if __name__ == '__main__':
	args = get_argparse()
# display a friendly message to the user
	print("Hi there {}, it's nice to meet you!".format(args["name"]))
	print(f'And you are in {args["place"]}')