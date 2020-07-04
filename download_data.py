# -*- coding: utf-8 -*-

import time
import requests
import os
from multiprocessing.pool import ThreadPool
from bs4 import BeautifulSoup
import wget
import zipfile
from tqdm import tqdm
import yaml

def get_tags_from_page(url,tag):
    """Extract a webpage and filter tag.

       Args:
            url {str} -- URL of webpage
            tag {str} -- tag to be filtered
       returns: 
            {list} -- tag mentioned
       """
    wp = requests.get(url)
    soup = BeautifulSoup(wp.text)
    return soup.find_all(tag)

def get_links_from_tags(url,tags, suffix):
    """Extract links with specific suffix from

    Args: 
        url {str} -- to be appended before link
        tags {list} -- list of tags containing links
        suffix {str} -- suffix to be downloaded
    Returns:
        {list} -- links
    """
    lst = list()
    for i in tags:
        if i["href"].endswith(suffix):
            lst.append(os.path.join(url,i["href"]))
    return lst

def unzip(path,suff,parallel,del_zip=True,out=None,print_time=True,processes=8):
    """Unzip all files with specified suffices

    Args:
        path {str} -- relative path to files
        suff {str} -- suffix
    Returns:
        None """
    def parallely(arg):
        path = arg
        handle = zipfile.ZipFile(path)
        handle.extractall(out)

    abs_path = os.path.abspath(path)
    if out is None:
        out = abs_path

    lst = list()
    for i in os.listdir(abs_path):
        if i.endswith(suff):
            lst.append(os.path.join(abs_path,i))

    start = time.time()
    if parallel:
        results = ThreadPool(processes).imap_unordered(parallely, lst)
        for i in tqdm(results):
            pass
    else:
        for j in tqdm(lst):
            handle = zipfile.ZipFile(j)
            handle.extractall(out)
    end = time.time()

    if print_time:
        print("Unzip Time: {:.2f} min.".format((end-start)/60))
    if del_zip:
        for i in lst:
            os.remove(i)

def download(links ,path, parallel ,print_time=True, processes=8):
    """ Download files from links

    Args:
        links {list} -- links of files to be downloaded
        path {str} -- relative path of output
        print_time {bool} -- prints download time if True
        parallel {bool} -- whether to use multiple threads
        processes {int} -- number of threads
    Returns: None
        """
    def parallely(arg):
        url,out = arg
        wget.download(url,out)

    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    path_list = [abs_path for i in range(len(links))]

    start = time.time()
    if parallel:
        args = list(zip(links,path_list))
        #print(args)
        results = ThreadPool(processes).imap_unordered(parallely, args)
        results = tqdm(results)
        for i in results:
            pass
    else:
        for i in tqdm(links):
            wget.download(i,abs_path)
    end = time.time()

    if print_time:
        print("Download Time: {:.2f} min.".format((end-start)/60))

def download_skyf_data():

    with open("config.yaml") as f:
        config = yaml.load(f)

    images_dir = config["path"]["images"]
    masks_dir = config["path"]["masks"]
    root = config["path"]["root"]

    skyfinder_url = config["url"]["skyfinder"]
    metadata_url = config["url"]["metadata"]
    tags = get_tags_from_page(config["url"]["skyfinder"], "a")
    zip_links = get_links_from_tags(skyfinder_url, tags, ".zip")
    mask_links = get_links_from_tags(skyfinder_url, tags, ".png")

    download(zip_links, images_dir, True)
    unzip(images_dir, ".zip", True)
    download(mask_links, masks_dir, True)
    download([metadata_url], root, False)

