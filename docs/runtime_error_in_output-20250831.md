(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# git reset 9c0b372c99c9f45add98992a90f11de0b2ce4132
Unstaged changes after reset:
M       scripts/run_prerl.sh
M       seg-r1/local_scripts/zero3_offload.json
M       seg-r1/src/open_r1/grpo_prerl.py
M       seg-r1/src/open_r1/trainer/grpo_trainer.py
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# git st
HEAD detached from 8b3363f
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   scripts/run_prerl.sh
        modified:   seg-r1/local_scripts/zero3_offload.json
        modified:   seg-r1/src/open_r1/grpo_prerl.py
        modified:   seg-r1/src/open_r1/trainer/grpo_trainer.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .cache/
        .hf_cache/
        .tmp/
        .triton/
        =1.5.2
        third_party/

no changes added to commit (use "git add" and/or "git commit -a")
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# git co .
Updated 4 paths from the index
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# bash scripts/run_prerl.sh 
[2025-09-01 08:52:21,885] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 09-01 08:52:22 __init__.py:190] Automatically detected platform cuda.
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:00<00:00, 513630.17it/s]
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.94it/s]
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Network error (HTTPError), entering retry loop.
^CW0901 08:53:45.858000 47674 site-packages/torch/distributed/elastic/agent/server/api.py:704] Received 2 death signal, shutting down workers
W0901 08:53:45.858000 47674 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 47709 closing signal SIGINT
Traceback (most recent call last):
  File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 393, in <module>
    main(script_args, training_args, model_args)
  File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 386, in main
    trainer.train()
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2241, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2455, in _inner_training_loop
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 507, in on_train_begin
    return self.call_event("on_train_begin", args, state, control)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 557, in call_event
    result = getattr(callback, event)(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 916, in on_train_begin
    self.setup(args, state, model, **kwargs)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 843, in setup
    self._wandb.init(
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 1252, in init
    return wi.init()
           ^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 815, in init
    result = run_init_handle.wait(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/lib/mailbox.py", line 283, in wait
    found, abandoned = self._slot._get_and_clear(timeout=wait_timeout)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/lib/mailbox.py", line 130, in _get_and_clear
    if self._wait(timeout=timeout):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/lib/mailbox.py", line 126, in _wait
    return self._event.wait(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/threading.py", line 629, in wait
    signaled = self._cond.wait(timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/threading.py", line 331, in wait
    gotit = waiter.acquire(True, timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 393, in <module>
[rank0]:     main(script_args, training_args, model_args)
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 386, in main
[rank0]:     trainer.train()
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2241, in train
[rank0]:     return inner_training_loop(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2455, in _inner_training_loop
[rank0]:     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 507, in on_train_begin
[rank0]:     return self.call_event("on_train_begin", args, state, control)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 557, in call_event
[rank0]:     result = getattr(callback, event)(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 916, in on_train_begin
[rank0]:     self.setup(args, state, model, **kwargs)
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 843, in setup
[rank0]:     self._wandb.init(
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 1252, in init
[rank0]:     return wi.init()
[rank0]:            ^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 815, in init
[rank0]:     result = run_init_handle.wait(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/lib/mailbox.py", line 283, in wait
[rank0]:     found, abandoned = self._slot._get_and_clear(timeout=wait_timeout)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/lib/mailbox.py", line 130, in _get_and_clear
[rank0]:     if self._wait(timeout=timeout):
[rank0]:        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/lib/mailbox.py", line 126, in _wait
[rank0]:     return self._event.wait(timeout=timeout)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/threading.py", line 629, in wait
[rank0]:     signaled = self._cond.wait(timeout)
[rank0]:                ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/threading.py", line 331, in wait
[rank0]:     gotit = waiter.acquire(True, timeout)
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: KeyboardInterrupt


^CW0901 08:54:13.860000 47674 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 47709 closing signal SIGTERM
Traceback (most recent call last):
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/agent/server/api.py", line 696, in run
    result = self._invoke_run(role)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/agent/server/api.py", line 855, in _invoke_run
    time.sleep(monitor_interval)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 84, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 47674 got signal: 2

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/autodl-tmp/envs/seg-r1/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 260, in launch_agent
    result = agent.run()
             ^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/metrics/api.py", line 137, in wrapper
    result = f(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/agent/server/api.py", line 705, in run
    self._shutdown(e.sigval)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py", line 365, in _shutdown
    self._pcontext.close(death_sig)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 572, in close
    self._close(death_sig=death_sig, timeout=timeout)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 909, in _close
    handler.proc.wait(time_to_wait)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/subprocess.py", line 1264, in wait
    return self._wait(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/subprocess.py", line 2047, in _wait
    time.sleep(delay)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 84, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 47674 got signal: 2
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# source /etc/network_turbo 
设置成功
注意：仅限于学术用途，不承诺稳定性保证
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# bash scripts/run_prerl.sh 
[2025-09-01 08:54:33,126] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 09-01 08:54:34 __init__.py:190] Automatically detected platform cuda.
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:00<00:00, 535056.00it/s]
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.97it/s]
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Network error (HTTPError), entering retry loop.
wandb: ERROR Run initialization has timed out after 90.0 sec. 
wandb: ERROR Please refer to the documentation for additional information: https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
Traceback (most recent call last):
  File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 393, in <module>
    main(script_args, training_args, model_args)
  File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 386, in main
    trainer.train()
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2241, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2455, in _inner_training_loop
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 507, in on_train_begin
    return self.call_event("on_train_begin", args, state, control)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 557, in call_event
    result = getattr(callback, event)(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 916, in on_train_begin
    self.setup(args, state, model, **kwargs)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 843, in setup
    self._wandb.init(
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 1266, in init
    wandb._sentry.reraise(e)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/analytics/sentry.py", line 155, in reraise
    raise exc.with_traceback(sys.exc_info()[2])
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 1252, in init
    return wi.init()
           ^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 844, in init
    raise error
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec. 
Please refer to the documentation for additional information: https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 393, in <module>
[rank0]:     main(script_args, training_args, model_args)
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 386, in main
[rank0]:     trainer.train()
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2241, in train
[rank0]:     return inner_training_loop(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2455, in _inner_training_loop
[rank0]:     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 507, in on_train_begin
[rank0]:     return self.call_event("on_train_begin", args, state, control)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer_callback.py", line 557, in call_event
[rank0]:     result = getattr(callback, event)(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 916, in on_train_begin
[rank0]:     self.setup(args, state, model, **kwargs)
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 843, in setup
[rank0]:     self._wandb.init(
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 1266, in init
[rank0]:     wandb._sentry.reraise(e)
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/analytics/sentry.py", line 155, in reraise
[rank0]:     raise exc.with_traceback(sys.exc_info()[2])
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 1252, in init
[rank0]:     return wi.init()
[rank0]:            ^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/wandb/sdk/wandb_init.py", line 844, in init
[rank0]:     raise error
[rank0]: wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec. 
[rank0]: Please refer to the documentation for additional information: https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
[rank0]:[W901 08:56:24.779582517 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
E0901 08:56:25.237000 48826 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 48883) of binary: /root/autodl-tmp/envs/seg-r1/bin/python3.11
Traceback (most recent call last):
  File "/root/autodl-tmp/envs/seg-r1/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
seg-r1/src/open_r1/grpo_prerl.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-09-01_08:56:25
  host      : autodl-container-d7654382a6-9240b673
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 48883)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# bash scripts/run_prerl.sh 
 
[2025-09-01 09:00:11,136] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 09-01 09:00:12 __init__.py:190] Automatically detected platform cuda.
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Resolving data files: 100%|█████████████████████████████████████| 3000/3000 [00:00<00:00, 531912.07it/s]
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Loading checkpoint shards: 100%|██████████████████████████████████████████| 5/5 [00:01<00:00,  4.99it/s]
  0%|                                                                           | 0/375 [00:00<?, ?it/s][rank0]: Traceback (most recent call last):
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 393, in <module>
[rank0]:     main(script_args, training_args, model_args)
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 386, in main
[rank0]:     trainer.train()
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2241, in train
[rank0]:     return inner_training_loop(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2548, in _inner_training_loop
[rank0]:     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 3698, in training_step
[rank0]:     loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/trainer/grpo_trainer.py", line 386, in compute_loss
[rank0]:     prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
[rank0]:                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/generation/utils.py", line 2223, in generate
[rank0]:     result = self._sample(
[rank0]:              ^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/generation/utils.py", line 3211, in _sample
[rank0]:     outputs = self(**model_inputs, return_dict=True)
[rank0]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/accelerate/utils/operations.py", line 818, in forward
[rank0]:     return model_forward(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/accelerate/utils/operations.py", line 806, in __call__
[rank0]:     return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank0]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 1795, in forward
[rank0]:     image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 554, in forward
[rank0]:     hidden_states = self._gradient_checkpointing_func(
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/_compile.py", line 32, in inner
[rank0]:     return disable_fn(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 632, in _fn
[rank0]:     return fn(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 489, in checkpoint
[rank0]:     return CheckpointFunction.apply(function, preserve, *args)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
[rank0]:     return super().apply(*args, **kwargs)  # type: ignore[misc]
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 264, in forward
[rank0]:     outputs = run_function(*args)
[rank0]:               ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 351, in forward
[rank0]:     hidden_states = hidden_states + self.attn(
[rank0]:                                     ^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 201, in forward
[rank0]:     q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 168, in apply_rotary_pos_emb_flashatt
[rank0]:     q_embed = apply_rotary_emb(q.float(), cos, sin).type_as(q)
[rank0]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/flash_attn/layers/rotary.py", line 121, in apply_rotary_emb
[rank0]:     return ApplyRotaryEmb.apply(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
[rank0]:     return super().apply(*args, **kwargs)  # type: ignore[misc]
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/flash_attn/layers/rotary.py", line 51, in forward
[rank0]:     out = apply_rotary(
[rank0]:           ^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/flash_attn/ops/triton/rotary.py", line 159, in apply_rotary
[rank0]:     torch.library.wrap_triton(rotary_kernel)[grid](
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: AttributeError: module 'torch.library' has no attribute 'wrap_triton'
  0%|                                                                           | 0/375 [00:12<?, ?it/s]
[rank0]:[W901 09:00:36.265881193 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
E0901 09:00:37.619000 50979 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 51003) of binary: /root/autodl-tmp/envs/seg-r1/bin/python3.11
Traceback (most recent call last):
  File "/root/autodl-tmp/envs/seg-r1/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
seg-r1/src/open_r1/grpo_prerl.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-09-01_09:00:37
  host      : autodl-container-d7654382a6-9240b673
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 51003)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# git log
commit 9c0b372c99c9f45add98992a90f11de0b2ce4132 (HEAD)
Author: donchiu <donchiu@redshore.space>
Date:   Sun Aug 31 16:33:47 2025 +0800

    utils/mddecathlon_to_segr1.py

commit 8413f81cf8ea1a32b92a932821d28886e0fe6170
Merge: 9a8882e 5dbb987
Author: Dong Chiu <donchiu@meta.com>
Date:   Sat Aug 30 14:23:03 2025 -0700

    Merge remote-tracking branch 'mine/singleGPU' into singleGPU

commit 9a8882e8a9a1fae4ba9d8c53b1a9c48067575803
Author: Dong Chiu <donchiu@meta.com>
Date:   Sat Aug 30 14:21:19 2025 -0700

    + docs/MedSAM2_inference_CT_Lesion.ipynb

commit 5dbb9878ddde766941da3d20c58087eba8d961f3
Author: donchiu <donchiu@redshore.space>
Date:   Sat Aug 30 16:06:30 2025 +0800

    A possible common problem to be resolved

commit f46563f792b0b8407215586c366a97cfb1e82c44
Author: Dong Chiu <donchiu@meta.com>
Date:   Sat Aug 30 01:04:14 2025 -0700

    + Cursor answer

commit 26a96184cd4201ea670d06a840df612a9c932295
Author: Dong Chiu <donchiu@meta.com>
Date:   Fri Aug 29 23:25:31 2025 -0700

    For single GPU

commit 29a04741c4dc62349b19a3b7eb6519e834570e17 (mine/debug, debug
)
Author: donchiu <donchiu@redshore.space>
Date:   Fri Aug 29 21:46:10 2025 +0800
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# bash scripts/run_prerl.sh 
 
[2025-09-01 09:06:18,527] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 09-01 09:06:19 __init__.py:190] Automatically detected platform cuda.
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Resolving data files: 100%|██████████████████████████████████████| 3000/3000 [00:00<00:00, 23238.94it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████| 5/5 [00:00<00:00,  5.79it/s]
  0%|                                                                           | 0/375 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 393, in <module>
[rank0]:     main(script_args, training_args, model_args)
[rank0]:   File "/root/autodl-tmp/works/Seg-R1/seg-r1/src/open_r1/grpo_prerl.py", line 386, in main
[rank0]:     trainer.train()
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2241, in train
[rank0]:     return inner_training_loop(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 2548, in _inner_training_loop
[rank0]:     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/trainer.py", line 3740, in training_step
[rank0]:     self.accelerator.backward(loss, **kwargs)
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/accelerate/accelerator.py", line 2734, in backward
[rank0]:     loss.backward(**kwargs)
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/_tensor.py", line 581, in backward
[rank0]:     torch.autograd.backward(
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/autograd/__init__.py", line 347, in backward
[rank0]:     _engine_run_backward(
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
[rank0]:     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/autograd/function.py", line 307, in apply
[rank0]:     return user_fn(self, *args)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 304, in backward
[rank0]:     outputs = ctx.run_function(*detached_inputs)
[rank0]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 351, in forward
[rank0]:     hidden_states = hidden_states + self.attn(
[rank0]:                                     ^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 320, in forward
[rank0]:     attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
[rank0]:                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.03 GiB. GPU 0 has a total capacity of 47.50 GiB of which 748.56 MiB is free. Process 137263 has 46.77 GiB memory in use. Of the allocated memory 45.13 GiB is allocated by PyTorch, and 1.16 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
  0%|                                                                           | 0/375 [00:23<?, ?it/s]
[rank0]:[W901 09:06:55.171030829 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
E0901 09:06:56.981000 53283 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 53318) of binary: /root/autodl-tmp/envs/seg-r1/bin/python3.11
Traceback (most recent call last):
  File "/root/autodl-tmp/envs/seg-r1/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/envs/seg-r1/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
seg-r1/src/open_r1/grpo_prerl.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-09-01_09:06:56
  host      : autodl-container-d7654382a6-9240b673
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 53318)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 
(seg-r1) root@autodl-container-d7654382a6-9240b673:~/autodl-tmp/works/Seg-R1# 