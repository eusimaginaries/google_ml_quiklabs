import argparse
import model
# import trainer.model as model
import tensorflow as tf
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--traindata',
      help='Training data file(s)',
      required=True
  )
  # parse args
  args = parser.parse_args()
  arguments = args.__dict__
  traindata = arguments.pop('traindata')

# Call read_dataset from model.py
feats, label = model.read_dataset(traindata)
# Find the average of all the labels that were read in
avg = tf.reduce_mean(label)
print(avg)
