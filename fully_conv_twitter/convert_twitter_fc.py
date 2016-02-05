import sys
import caffe


SUBSETS = ['test1','test2','test3','test4','test5']

def convert_twitter_weights_to_fully_conv(original_deploy, original_caffemodel, fc_deploy, fc_caffemodel_save_path):
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net(original_deploy, original_caffemodel, caffe.TEST)
    params = ['fc6', 'fc7', 'fc8_twitter']
    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(fc_deploy, original_caffemodel, caffe.TEST)
    params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8_twitter-conv']
    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

    # Trasplant weights
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    # Save weights (caffemodel) for the fc net
    net_full_conv.save(fc_caffemodel_save_path)


if (len(sys.argv)>=2):
    try:
        model = str(sys.argv[1])
    except:
        sys.exit('The given arguments are not correct')
else:
    sys.exit("Not enough arguments. Run 'python convert_twitter_fc.py model'")


if model == 'imagenet':
    model_folder = '5-fold_cross-validation'
elif model == 'places':
    MEAN_FILE = '/imatge/vcampos/caffe/models/places/places205CNN_mean.npy'
    model_folder = 'places_5-fold_CV'
elif model == 'deepsentibank':
    model_folder = 'deepsentibank_5-fold_CV'
elif model == 'mvso_en' or model == 'mvso_sp' or model == 'mvso_fr' or model == 'mvso_it' or model == 'mvso_ch' or model=='mvso_ge':
    model_folder = model + '_5-fold_CV'
else:
    sys.exit("The requested model is not valid")


for subset in SUBSETS:
    # Define paths
    original_deploy = '/imatge/vcampos/work/twitter_finetuning/'+model_folder+'/'+subset+'/deploy.prototxt'
    original_caffemodel = '/imatge/vcampos/work/twitter_finetuning/'+model_folder+'/trained/twitter_finetuned_'+subset+'_iter_180.caffemodel'
    fc_deploy = '/imatge/vcampos/work/twitter_finetuning/'+model_folder+'/'+subset+'/deploy_conv.prototxt'
    fc_caffemodel_save_path = '/imatge/vcampos/work/twitter_finetuning/'+model_folder+'/trained/twitter_finetuned_'+subset+'_iter_180_conv.caffemodel'

    # Perform conversion
    convert_twitter_weights_to_fully_conv(original_deploy=original_deploy,
                                          original_caffemodel=original_caffemodel,
                                          fc_deploy=fc_deploy,
                                          fc_caffemodel_save_path=fc_caffemodel_save_path)
