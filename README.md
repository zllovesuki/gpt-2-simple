# gpt-2-simple with validation

Validation function from Neil Shepperd's [GPT-2 package](https://github.com/nshepperd/gpt-2) added to Max Woolf's [GPT-2 package](https://github.com/minimaxir/gpt-2-simple).

## Usage

    import gpt_2_simple as gpt2
    import os
    import requests
    
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)
    
    train_file = "20K_train.txt"
    test_file = "20K_test.txt"
    
    sess = gpt2.start_tf_sess()

    gpt2.finetune(sess,
                  dataset = train_file,
                  model_name = model_name,
                  run_name = '124_valid',
                  steps = 1000,
                  print_every = 100,
                  batch_size = 1,
                  learning_rate = 0.00001,
                  val_dataset = test_file,
                  val_batch_size = 2,
                  val_batch_count = 50,
                  val_every = 100
                  )

## Results
![](loss_before_validation.png)
![](loss_after_validation.png)
