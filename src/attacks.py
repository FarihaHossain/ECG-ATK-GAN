pip install numpy>=1.17.2
pip install argparse>=1.1
pip install pandas>=0.25.1
pip install scikit-learn>=0.21.3
pip install matplotlib==3.1.3
pip install adversarial-robustness-toolbox==1.7.2 keras


# define the standalone discriminator model
def define_discriminator(in_shape=(280,1), n_classes=4):

    in_image = Input(shape=(in_shape))    
    fe = Conv1D(32, 3, strides=1, padding='same')(in_image)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(32, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(64, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(128, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(128, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(256, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(256, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    out2 = Dense(n_classes, activation='softmax')(fe)
    model = Model(inputs=in_image, outputs=out2,name="Discriminator")
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    #model.compile(loss=['categorical_crossentropy'], optimizer=opt)
    #model.summary()
    return model
	

#FGSM

tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, CarliniLInfMethod, CarliniL2Method,BasicIterativeMethod, ProjectedGradientDescent

## Cant do Pixelattack, thresholdattack, squareattack, ShadowAttack
# CW inf takes long time, CW L2 takes even more time
# Boundary attack too slow 2-3 hours for the parameters given below on X_test
model = define_discriminator()
classifier = KerasClassifier(model=model, clip_values=(np.amin(X_train), np.amax(X_train)))
#X_test_3d = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_train_3d = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))	

#epsilon =0.1
attack_fgsm_e1 = FastGradientMethod(estimator=classifier, eps=0.01)
x_train_adv_e1 = attack_fgsm_e1.generate(X_train_3d)

np.savez('FGSM/x_train_adv_e1', x=x_train_adv_e1, y=Y_train)

#epsilon =0.1
attack_fgsm_e1 = FastGradientMethod(estimator=classifier, eps=0.01)
x_test_adv_e1 = attack_fgsm_e1.generate(X_test_3d)

#PGD

classifier = KerasClassifier(model=model, clip_values=(np.amin(X_train), np.amax(X_train)))
#X_test_3d = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_train_3d = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#epsilon =0.1
attack_pgd_e1 = ProjectedGradientDescent(estimator=classifier,  eps=0.01,batch_size=10000)
x_train_adv_pgd_e1 = attack_pgd_e1.generate(X_train_3d)

np.savez('PGD/x_train_adv_e1', x=x_train_adv_pgd_e1, y=Y_train)
classifier = KerasClassifier(model=model, clip_values=(np.amin(X_test), np.amax(X_test)))
#X_test_3d = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_test_3d = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

x_test_adv_pgd_e1 = attack_pgd_e1.generate(X_test_3d)

np.savez('PGD/x_test_adv_e1', x=x_test_adv_pgd_e1, y=Y_test)


#CW
#epsilon =0.1
attack_cw_e1 = CarliniLInfMethod(classifier=classifier, eps=0.1, max_iter=10,batch_size=10000, learning_rate=0.01)
x_train_adv_cw_e1 = attack_cw_e1.generate(X_train_3d)
np.savez('/content/drive/MyDrive/MICCAI2022_ptbdb/CW/x_train_adv_cw_e1', x=x_train_adv_cw_e1, y=Y_train)

x_test_adv_cw_e1 = attack_cw_e1.generate(X_test_3d)
np.savez('CW/x_test_adv_cw_e1', x=x_test_adv_cw_e1, y=Y_test)

#BIM
#epsilon =0.1
attack_bim_e1 = BasicIterativeMethod(estimator=classifier,  eps=0.01,batch_size=10000)
x_train_adv_bim_e1 = attack_bim_e1.generate(X_train_3d)
np.savez('BIM/x_train_adv_bim_e1', x=x_train_adv_bim_e1, y=Y_train)

x_test_adv_bim_e1 = attack_bim_e1.generate(X_test_3d)
np.savez('BIM/x_test_adv_bim_e1', x=x_test_adv_bim_e1, y=Y_test)


#Boundary Box
tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier

from art.attacks.evasion import BoundaryAttack
model = define_discriminator()
classifier = KerasClassifier(model=model, clip_values=(np.amin(X_train), np.amax(X_train)))

BoundaryAttack_untargeted  = BoundaryAttack(estimator=classifier,
                                            batch_size = 8000,
                                            max_iter=0,
                                            targeted= False,
                                            num_trial= 1,
                                            sample_size = 8000,
                                            init_size = 1,
                                            verbose=True)
X_train_3d = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

X_train_attacked_ptb = BoundaryAttack_untargeted.generate(X_train_3d)

np.savez('Boundary_untargeted/X_train_attacked_ptb', x=X_train_attacked_ptb, y=Y_train)

#HSJ

from art.attacks.evasion import HopSkipJump
model = define_discriminator()
classifier = KerasClassifier(model=model, clip_values=(np.amin(X_train), np.amax(X_train)))

HopSkipJump_train = HopSkipJump(classifier=classifier,batch_size= 10000,max_iter = 0,max_eval=1000,init_eval=10, verbose = True)

X_train_3d = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
HopSkipJump_train_D = HopSkipJump_train.generate(X_train_3d)

np.savez('Train_attacked', x=HopSkipJump_train_D, y=Y_train)
