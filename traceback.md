Traceback (most recent call last):
  File "/home/apex/code/Evo-RL-data/convert_pkl_dataset.py", line 458, in ssh_python
    result = subprocess.run(
  File "/home/apex/code/evo-rl/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['ssh', '-o', 'BatchMode=yes', '-p', '2222', 'baiyuntian@10.61.5.3', 'python3 -c \'import hashlib,json,pathlib,sys\ndef verify_directory(root, expected):\n    if not root.is_dir() or root.is_symlink():\n        raise ValueError(\'"\'"\'Upload destination is not a regular directory: \'"\'"\'+str(root))\n    paths=list(root.rglob(\'"\'"\'*\'"\'"\'))\n    if any(p.is_symlink() for p in paths):\n        raise ValueError(\'"\'"\'Unexpected symlink in upload: \'"\'"\'+str(root))\n    actual={p.relative_to(root).as_posix() for p in paths if p.is_file()}\n    if actual != set(expected)|{\'"\'"\'upload.sha256.json\'"\'"\'}:\n        raise ValueError(\'"\'"\'Remote file list mismatch\'"\'"\')\n    uploaded=json.loads((root/\'"\'"\'upload.sha256.json\'"\'"\').read_text(encoding=\'"\'"\'utf-8\'"\'"\'))\n    if uploaded != expected:\n        raise ValueError(\'"\'"\'Remote manifest differs from fresh local manifest\'"\'"\')\n    for name,wanted in expected.items():\n        relative=pathlib.PurePosixPath(name)\n        if relative.is_absolute() or \'"\'"\'..\'"\'"\' in relative.parts:\n            raise ValueError(\'"\'"\'Invalid manifest path: \'"\'"\'+name)\n        p=root/name\n        if not p.resolve().is_relative_to(root.resolve()):\n            raise ValueError(\'"\'"\'Invalid manifest path: \'"\'"\'+name)\n        digest=hashlib.sha256()\n        with p.open(\'"\'"\'rb\'"\'"\') as f:\n            for chunk in iter(lambda:f.read(4194304),b\'"\'"\'\'"\'"\'): digest.update(chunk)\n        if digest.hexdigest()!=wanted:\n            raise ValueError(\'"\'"\'Remote SHA256 mismatch: \'"\'"\'+name)\n\nparent,temporary,final=map(pathlib.Path,sys.argv[1:])\nexpected=json.load(sys.stdin)\nparent.mkdir(parents=True,exist_ok=True)\nif final.exists() or final.is_symlink():\n    if final.is_dir() and not final.is_symlink() and not any(final.iterdir()):\n        print(json.dumps({\'"\'"\'status\'"\'"\':\'"\'"\'ready\'"\'"\'}))\n    else:\n        verify_directory(final,expected)\n        print(json.dumps({\'"\'"\'status\'"\'"\':\'"\'"\'already_complete\'"\'"\',\'"\'"\'path\'"\'"\':str(final)}))\nelse:\n    if temporary.exists() or temporary.is_symlink():\n        raise FileExistsError(\'"\'"\'Remote temporary destination already exists\'"\'"\')\n    print(json.dumps({\'"\'"\'status\'"\'"\':\'"\'"\'ready\'"\'"\'}))\n\' /nfs/baiyuntian/assets/caiji /nfs/baiyuntian/assets/caiji/test.uploading-a959118c0ece /nfs/baiyuntian/assets/caiji/test']' returned non-zero exit status 255.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/apex/code/Evo-RL-data/convert_pkl_dataset.py", line 555, in <module>
    main()
  File "/home/apex/code/Evo-RL-data/convert_pkl_dataset.py", line 551, in main
    upload(output, host, user, remote, port)
  File "/home/apex/code/Evo-RL-data/convert_pkl_dataset.py", line 468, in upload
    preparation = ssh_python(
  File "/home/apex/code/Evo-RL-data/convert_pkl_dataset.py", line 465, in ssh_python
    raise RuntimeError(f"Remote upload step failed: {detail}") from error
RuntimeError: Remote upload step failed: Host key verification failed.