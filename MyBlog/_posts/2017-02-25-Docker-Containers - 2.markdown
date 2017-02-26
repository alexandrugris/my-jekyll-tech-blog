---
layout: post
title:  "Docker And Containers Part 2"
date:   2017-02-25 10:25:16 +0200
categories: Web Development
---
This is the second part of my personal Docker cheatsheet. So why Docker? Because I am interested in distributed computing and I want to play with various technologies. 
I want to be able to spin quickly complex setups and, at the same time, keep my desktop as clean as possible. 

Here is the first part: [Docker And Containers]({% post_url 2017-02-12-Docker-Containers %}) 

### Using host files in a docker containers

First thing, we need to enable drives to be visible within the container. For this, we use the Settings dialog from Docker For Windows.

![Settings Dialog Mount FileSystem]({{site.url}}/assets/docker_2_1.png)

Port 445 must be opened in the Windows Firewall.

![Firewall warning]({{site.url}}/assets/docker_2_2.png)

> To share the drive, allow connections between the Windows host machine and the virtual machine in Windows Firewall or your third party firewall software. You do not need to open port 445 on any other network. 
> By default, allow connections to 10.0.75.1 port 445 (the Windows host) from 10.0.75.2 (the virtual machine). 

*Attention:* 

By default, the Docker network is considered in the Public Networks profile. 
For my desktop PC at home, I have simply enabled firewall rules for the public networks (default is to block all inbound connections). This setting might be dangerous on a laptop.

![Network rules]({{site.url}}/assets/docker_2_3.png)

Another option is to run `gpedit.msc` and change network profile for *Unidentifiable Networks*. :)

Now that we have enabled access to host folders, let's run the following command:

```
c:>docker run --rm -v d:\OneDrive\Pictures:/my_pictures alpine sh -c "cd /my_pictures && ls -l"
```

![Running a command on a volume in docker]({{site.url}}/assets/docker_2_4.png)

Let's disect it:

1. `docker run --rm` tells docker to remove the image after the command is run. The image is not persisted. 
2. `-v d:\OneDrive\Pictures:/my_pictures` tells docker to mount the `d:\OneDrive\Pictures` from the host to the `/my_pictures` folder in the container.
3. `alpine sh -c "cd /my_pictures && ls -l"` from the `alpine` image run the `sh` command with the `-c "cd /my_pictures && ls -l"` parameters. It means change directory to `/my_pictures` and then, if successful, run the `ls -l` command.

For a list of parameters of the `docker run` command, here they are: [Docker Run](https://docs.docker.com/engine/reference/commandline/run/). Another parameter very useful is the `--env` to sent environment variables in the container.

### Example no. 1: delete all "mongo"-named containers:

The following example uses a mixture of batch commands (running on the Windows host) and Linux commands running in the Docker container.

```
docker ps -a 
    > d:\to_delete.txt 
    && docker run --rm -v d:\to_delete.txt:/my_data/to_delete.txt 
    alpine sh -c "grep mongo /my_data/to_delete.txt | awk '{print $1}'" 
    > d:\to_delete_2.txt 
    && for /F %I in (d:\to_delete_2.txt) DO docker rm %I
```

Explanation: the command uses two files `d:\to_delete.txt` and `d:\to_delete_2.txt` as IPC between Windows and the command ran in the Linux container.


### <a name="vclinuxdocker"></a> Example no. 2: developing Linux applications with Visual C++ for Linux and Docker

Visual C++ for Linux is great and  works right outside the box when connected to a Virtual Box VM (in the picture below the VM is running a Debian). 

![VM VC for Linux]({{site.url}}/assets/vsdebian.jpg)

However, creating full VMs, starting them and moving them around is complex and tedious. Here is a ligher alternative - use a Docker container for running and debugging the app.

1. Run a Debian Linux distribution. Map ports. Create a local user and install g++, gdb and openssh-server. Drop into the bash console. Attention to the flag `--security-opt seccomp=unconfined` - gdb will not be able to connect to the executable otherwise.

```
docker run -it --rm -p 2222:22 --security-opt seccomp=unconfined
    debian bash -c "echo sshd: ALL >> /etc/hosts.allow 
        && apt-get update 
        && apt-get install -y openssh-server g++ gdb gdbserver 
        && mkdir /home/agris 
        && useradd -d /home/agris agris 
        && passwd agris 
        && /etc/init.d/ssh restart 
        && chown -hR agris /home/agris 
        && bash"
```

![Running the command above]({{site.url}}/assets/docker_2_8.png)

2. Type Linux password when prompted.

3. Create a Visual C++ for Linux project. Set connection to localhost on port 2222 (forwarded 22, the default sshd port, from the container)

![Visual Studio Project Config]({{site.url}}/assets/docker_2_5.png)

4. Set the debugger type to gdb. 

![Visual Studio Project Config]({{site.url}}/assets/docker_2_6.png)

5. Clean / Rebuild the project.

6. Happy debugging. :)

![Visual Studio Project Config]({{site.url}}/assets/docker_2_7.png)

The container is run with the `--rm` flag above. You may want to remove it not to install everything everytime the command is run. 