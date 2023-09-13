# from scan import detect, classifier
# from register import detect, preprocess

# if a > 2:
#     test_run_a.main()
# else:
#     test_run_b.main()

import os
a = int(input('enter:'))
if a > 2:
    os.system('python Project3/Save-Pets-ML-main/SVM-Classifier/Classifier_X.py --test test_0.jpg --dir Dog-Data')
else:
    print('no')