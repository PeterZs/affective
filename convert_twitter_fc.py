import CNN_tools


SUBSETS = ['test1','test2','test3','test4','test5']

for subset in SUBSETS:
    # Define paths
    original_deploy = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/'+subset+'/deploy.prototxt'
    original_caffemodel = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/trained/twitter_finetuned_'+subset+'_iter_180.caffemodel'
    fc_deploy = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/'+subset+'/deploy_conv.prototxt'
    fc_caffemodel_save_path = '/imatge/vcampos/work/twitter_finetuning/5-fold_cross-validation/trained/twitter_finetuned_'+subset+'_iter_180_conv.caffemodel'

    # Perform conversion
    CNN_tools.convert_twitter_weights_to_fully_conv(original_deploy=original_deploy,
                                                    original_caffemodel=original_caffemodel,
                                                    fc_deploy=fc_deploy,
                                                    fc_caffemodel_save_path=fc_caffemodel_save_path)
