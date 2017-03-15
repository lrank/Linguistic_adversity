import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import cPickle

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("dataset", "mr", "Data source (mr).")
tf.flags.DEFINE_string("positive_data_file", ".pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", ".neg", "Data source for the positive data.")
tf.flags.DEFINE_string("noise_type", "raw", "Noise type (raw)")
tf.flags.DEFINE_boolean("is_noise_train", False, "Noise training flag (False).")
tf.flags.DEFINE_boolean("is_noise_test", False, "Noise test flag (False).")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.noise_type != "raw":
    FLAGS.positive_data_file = "../data/" + FLAGS.dataset + FLAGS.positive_data_file + '.' + FLAGS.noise_type
    FLAGS.negative_data_file = "../data/" + FLAGS.dataset + FLAGS.negative_data_file + '.' + FLAGS.noise_type

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
#x_text [[],[]]
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
x_all = []
for doc in x_text:
    x_all.extend(doc)

max_document_length = max([len(x.split(" ")) for x in x_all])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_all)))


print ("loading word2vec vectors...")
Dict = data_helpers.load_bin_vec("/Data/GoogleNews-vectors-negative300.bin", vocab_processor.vocabulary_._mapping)
print ("word2vec loaded!")
print ("num words already in word2vec: " + str(len(Dict)) )
data_helpers.add_unknown_words(Dict, vocab_processor.vocabulary_._mapping)
cPickle.dump([Dict], open(FLAGS.dataset+".p."+FLAGS.noise_type, "wb") )
# exit()
tmp = cPickle.load( open(FLAGS.dataset+".p."+FLAGS.noise_type, "rb") )
Dict = tmp[0]
w2v = []
for word_number in range( len(Dict) ):
    w2v.append( Dict[vocab_processor.vocabulary_._reverse_mapping[word_number]].tolist() )


# Randomly shuffle data
np.random.seed(10)
x_shuffled = x_text
y_shuffled = y

cv_score = []
#Cross-validation
for cv in range(10):
    best_score_in_cv = 0

    x_train, x_dev = [], []
    y_train, y_dev = [], []
    # Split train/test set
    for i in range( len(x_shuffled) ):
        if i % 10 == cv:
            x_dev.append(x_shuffled[i])
            y_dev.append(y_shuffled[i])
        else:
            x_train.append(x_shuffled[i])
            y_train.append(y_shuffled[i])

    y_train = np.array(y_train)
    
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement,
          # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.24),
          intra_op_parallelism_threads=2,
          inter_op_parallelism_threads=2)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            sess.run(cnn.W.assign(w2v))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                best = 0
                if accuracy > best:
                    best = accuracy
                if writer:
                    writer.add_summary(summaries, step)
                return best

            #Noisify dev data
            x_dev_single = []
            if FLAGS.is_noise_test == True:
                print "Noisify test data"
                for doc in x_dev:
                    x_dev_single.append( doc[ np.random.randint(len(doc)) ])
            else:
                for doc in x_dev:
                    x_dev_single.append( doc[ 0 ])
            x_dev_single = np.array( list(vocab_processor.transform(x_dev_single)) )

            x_train_single = []
            for doc in x_train:
                x_train_single.append( doc[ 0 ] )
            x_train_single = np.array( list(vocab_processor.transform(x_train_single)) )

            for _ in range(FLAGS.num_epochs):
                #Noisify train data for each epoch
                if FLAGS.is_noise_train == True:
                    print "Noisify training data"
                    x_train_single = []
                    for doc in x_train:
                        x_train_single.append( doc[ np.random.randint(len(doc)) ] )
                    x_train_single = np.array( list(vocab_processor.transform(x_train_single)) )

                # Generate mini-batches
                batches = data_helpers.batch_iter(
                    list(zip(x_train_single, y_train)), FLAGS.batch_size, 1)
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_score = dev_step(x_dev_single, y_dev, writer=dev_summary_writer)
                        if dev_score > best_score_in_cv:
                            best_score_in_cv = dev_score
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
            cv_score.append( best_score_in_cv )


print cv_score
print np.average( np.array( cv_score ) )
