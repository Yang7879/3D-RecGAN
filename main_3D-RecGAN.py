import os
import shutil
import tensorflow as tf
import scipy.io
import tools

resolution = 64
batch_size = 8
GPU0 = '1'

###############################################################
config={}
config['train_names'] = ['chair']
for name in config['train_names']:
    config['X_train_'+name] = '/home/yang/Data/3D-RecGAN_Data/'+name+'/train_25d/voxel_grids_64/'
    config['Y_train_'+name] = '/home/yang/Data/3D-RecGAN_Data/'+name+'/train_3d/voxel_grids_64/'
config['test_names']=['chair']
for name in config['test_names']:
    config['X_test_'+name] = '/home/yang/Data/3D-RecGAN_Data/'+name+'/test_25d/voxel_grids_64/'
    config['Y_test_'+name] = '/home/yang/Data/3D-RecGAN_Data/'+name+'/test_3d/voxel_grids_64/'
config['resolution'] = resolution
config['batch_size'] = batch_size
################################################################

###############################################################
config_Dell={}
config_Dell['train_names'] = ['chair']
for name in config_Dell['train_names']:
    config_Dell['X_train_'+name] = '/media/disk5/yang/Data/3D-RecGAN_Data/'+name+'/train_25d/voxel_grids_64/'
    config_Dell['Y_train_'+name] = '/media/disk5/yang/Data/3D-RecGAN_Data/'+name+'/train_3d/voxel_grids_64/'
config_Dell['test_names']=['chair']
for name in config_Dell['test_names']:
    config_Dell['X_test_'+name] = '/media/disk5/yang/Data/3D-RecGAN_Data/'+name+'/test_25d/voxel_grids_64/'
    config_Dell['Y_test_'+name] = '/media/disk5/yang/Data/3D-RecGAN_Data/'+name+'/test_3d/voxel_grids_64/'
config_Dell['resolution'] = resolution
config_Dell['batch_size'] = batch_size
################################################################

class Network:
    def __init__(self):
        self.train_mod_dir = './train_mod/'
        self.train_sum_dir = './train_sum/'
        self.test_res_dir = './test_res/'
        self.test_sum_dir = './test_sum/'

        if os.path.exists(self.test_res_dir):
            shutil.rmtree(self.test_res_dir)
            print ('test_res_dir: deleted and then created!')
        os.makedirs(self.test_res_dir)

        if os.path.exists(self.train_mod_dir):
            shutil.rmtree(self.train_mod_dir)
            print ('train_mod_dir: deleted and then created!')
        os.makedirs(self.train_mod_dir)

        if os.path.exists(self.train_sum_dir):
            shutil.rmtree(self.train_sum_dir)
            print ('train_sum_dir: deleted and then created!')
        os.makedirs(self.train_sum_dir)

        if os.path.exists(self.test_sum_dir):
            shutil.rmtree(self.test_sum_dir)
            print ('test_sum_dir: deleted and then created!')
        os.makedirs(self.test_sum_dir)

    def ae_u(self,X):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[batch_size,resolution,resolution,resolution,1])
            ##### encode
            c_e = [1, 64, 128, 256, 512]
            s_e = [0, 1, 1, 1, 1]
            layers_e = []
            layers_e.append(X)
            for i in range(1, 5, 1):
                layer = tools.Ops.conv3d(layers_e[-1], k=4, out_c=c_e[i], str=s_e[i], name='e' + str(i))
                layer = tools.Ops.maxpool3d(tools.Ops.xxlu(layer,name='lrelu'), k=2, s=2, pad='SAME')
                layers_e.append(layer)

            ##### fc
            bat, d1, d2, d3, cc = [int(d) for d in layers_e[-1].get_shape()]
            lfc = tf.reshape(layers_e[-1], [bat, -1])
            lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=5000, name='fc1'), name='relu')

        with tf.device('/gpu:'+GPU0):
            lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=d1 * d2 * d3 * cc, name='fc2'),name='relu')
            lfc = tf.reshape(lfc, [bat, d1, d2, d3, cc])

            ##### decode
            c_d = [0, 256, 128, 64, 1]
            s_d = [0, 2, 2, 2, 2, 2]
            layers_d = []
            layers_d.append(lfc)
            for j in range(1, 5, 1):
                u_net = True
                if u_net:
                    layer = tf.concat([layers_d[-1], layers_e[-j]], axis=4)
                    layer = tools.Ops.deconv3d(layer, k=4, out_c=c_d[j], str=s_d[j], name='d' + str(len(layers_d)))
                else:
                    layer = tools.Ops.deconv3d(layers_d[-1], k=4, out_c=c_d[j], str=s_d[j], name='d' + str(len(layers_d)))

                if j != 4:
                    layer = tools.Ops.xxlu(layer,name='relu')
                layers_d.append(layer)

            vox_sig = tf.sigmoid(layers_d[-1])
            vox_sig_modified = tf.maximum(vox_sig,0.01)
        return vox_sig, vox_sig_modified

    def dis(self, X, Y):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[batch_size,resolution,resolution,resolution,1])
            Y = tf.reshape(Y,[batch_size,resolution,resolution,resolution,1])
            layer = tf.concat([X,Y],axis=4)
            c_d = [1,64,128,256,512]
            s_d = [0,2,2,2,2]
            layers_d =[]
            layers_d.append(layer)
            for i in range(1,5,1):
                layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d'+str(i))
                if i!=4:
                    layer = tools.Ops.xxlu(layer, name='lrelu')
                layers_d.append(layer)
            y = tf.reshape(layers_d[-1],[batch_size,-1])
        return tf.nn.sigmoid(y)

    def train(self, data):
        X = tf.placeholder(shape=[batch_size, resolution, resolution, resolution, 1], dtype=tf.float32)
        Y = tf.placeholder(shape=[batch_size, resolution, resolution, resolution, 1], dtype=tf.float32)
        lr = tf.placeholder(tf.float32)

        with tf.variable_scope('ae'):
            Y_pred, Y_pred_modi = self.ae_u(X)

        with tf.variable_scope('dis'):
            XY_real_pair = self.dis(X, Y)
        with tf.variable_scope('dis',reuse=True):
            XY_fake_pair = self.dis(X, Y_pred)

        with tf.device('/gpu:'+GPU0):
            ################################ ae loss
            Y_ = tf.reshape(Y,shape=[batch_size,-1])
            Y_pred_modi_ = tf.reshape(Y_pred_modi,shape=[batch_size,-1])
            w = 0.85
            ae_loss = tf.reduce_mean( -tf.reduce_mean(w*Y_*tf.log(Y_pred_modi_ + 1e-8),reduction_indices=[1]) -
                                      tf.reduce_mean((1-w)*(1-Y_)*tf.log(1-Y_pred_modi_ + 1e-8), reduction_indices=[1]) )
            sum_ae_loss = tf.summary.scalar('ae_loss', ae_loss)

            ################################ wgan loss
            gan_g_loss = -tf.reduce_mean(XY_fake_pair)
            gan_d_loss_no_gp = tf.reduce_mean(XY_fake_pair) - tf.reduce_mean(XY_real_pair)
            sum_gan_g_loss = tf.summary.scalar('gan_g_loss',gan_g_loss)
            sum_gan_d_loss_no_gp = tf.summary.scalar('gan_d_loss_no_gp',gan_d_loss_no_gp)
            alpha = tf.random_uniform(shape=[batch_size,resolution**3],minval=0.0,maxval=1.0)

            Y_pred_ = tf.reshape(Y_pred,shape=[batch_size,-1])
            differences_ = Y_pred_ -Y_
            interpolates = Y_ + alpha*differences_
            with tf.variable_scope('dis',reuse=True):
                XY_fake_intep = self.dis(X, interpolates)
            gradients = tf.gradients(XY_fake_intep,[interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
            sum_gp = tf.summary.scalar('wgan_gp', gradient_penalty)
            gan_d_loss_gp = gan_d_loss_no_gp+10*gradient_penalty

            #################################  ae + gan loss
            gan_g_w = 5
            ae_w = 100-gan_g_w
            ae_gan_g_loss = ae_w * ae_loss + gan_g_w * gan_g_loss

        with tf.device('/gpu:' + GPU0):
            ae_var = [var for var in tf.trainable_variables() if var.name.startswith('ae')]
            dis_var = [var for var in tf.trainable_variables() if var.name.startswith('dis')]
            ae_g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(ae_gan_g_loss, var_list=ae_var)
            dis_optim = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(gan_d_loss_gp,var_list=dis_var)

        print tools.Ops.variable_count()
        sum_merged = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        with tf.Session(config=config) as sess:
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

            if os.path.isfile(self.train_mod_dir + 'model.cptk.data-00000-of-00001'):
                print ('restoring saved model')
                saver.restore(sess, self.train_mod_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(15):
                data.shuffle_X_Y_files(label='train')
                total_train_batch_num = data.total_train_batch_num
                print ('total_train_batch_num:', total_train_batch_num)
                for i in range(total_train_batch_num):
                    #### training
                    X_train_batch, Y_train_batch = data.load_X_Y_voxel_grids_train_next_batch()
                    sess.run([dis_optim], feed_dict={X: X_train_batch, Y: Y_train_batch, lr: 0.0001})
                    sess.run([ae_g_optim], feed_dict={X: X_train_batch, Y: Y_train_batch, lr: 0.0005})
                    ae_loss_c,gan_g_loss_c,gan_d_loss_no_gp_c,gan_d_loss_gp_c,sum_train = sess.run(
                    [ae_loss, gan_g_loss, gan_d_loss_no_gp, gan_d_loss_gp,sum_merged],feed_dict={X: X_train_batch, Y: Y_train_batch})

                    if i%100==0:
                        sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
                    print ('epoch:', epoch, 'i:', i, 'train ae loss:', ae_loss_c,'gan g loss:', gan_g_loss_c,
                           'gan d loss no gp:',gan_d_loss_no_gp_c, 'gan d loss gp:', gan_d_loss_gp_c)

                    #### testing
                    if i % 300 == 0 and epoch % 1 == 0:
                        X_test_batch, Y_test_batch = data.load_X_Y_voxel_grids_test_next_batch(fix_sample=False)
                        ae_loss_t,gan_g_loss_t,gan_d_loss_no_gp_t,gan_d_loss_gp_t, Y_test_pred = sess.run(
                        [ae_loss, gan_g_loss, gan_d_loss_no_gp,gan_d_loss_gp, Y_pred],feed_dict={X: X_test_batch, Y: Y_test_batch})

                        to_save = {'X_test': X_test_batch, 'Y_test_pred': Y_test_pred,'Y_test_true': Y_test_batch}
                        scipy.io.savemat(self.test_res_dir + 'X_Y_pred_' + str(epoch).zfill(2) + '_' + str(i).zfill(4) + '.mat', to_save, do_compression=True)
                        print ('epoch:', epoch, 'i:', i, 'test ae loss:', ae_loss_t, 'gan g loss:',gan_g_loss_t,
                               'gan d loss no gp:', gan_d_loss_no_gp_t, 'gan d loss gp:', gan_d_loss_gp_t)

                    #### full testing
                    # ...

                    #### model saving
                    if i %500==0 and i>0 and epoch % 1 == 0:
                        saver.save(sess, save_path=self.train_mod_dir +'model.cptk')
                        print "epoch:", epoch, " i:", i, " model saved!"

if __name__ == "__main__":
    data = tools.Data(config)
    net = Network()
    net.train(data)