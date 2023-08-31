#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Storage handling utility
"""
import glob
import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import List, Optional
from zipfile import ZipFile

import boto3  # type: ignore


def stringlist_regex(stringlist: List[str], pattern: str) -> List[str]:
    """
    Filter product content with a regexp
    """
    return [f for f in stringlist if fnmatch(f, pattern)]


@dataclass(frozen=True)
class S3Context:
    """
    """
    resource: boto3.resources.factory.ResourceModel
    bucket: str


def agnostic_regex(product_dir: str,
                   pattern: str,
                   s3_context: Optional[S3Context] = None,
                   use_gdal_adressing: bool = False) -> List[str]:
    """
    Agnostic regex search, both s3 and posix, zipped or not.
    Optionally turns path to gdal compatible addresses
    """
    if s3_context is None:
        # Posix case
        if product_dir.endswith('.zip') or product_dir.endswith('.ZIP'):
            # Zipped case
            with ZipFile(product_dir) as zip_ds:
                files_list = zip_ds.namelist()
                matches = stringlist_regex(files_list, pattern)
                if use_gdal_adressing:
                    return [f'/vsizip/{product_dir}/{p}' for p in matches]
                return matches
        # Plain case
        matches = glob.glob(f"{product_dir}/{pattern}")
        if use_gdal_adressing:
            return matches
        return [m[len(product_dir) + 1:] for m in matches]
    # Now in S3 case
    if product_dir.endswith('.zip') or product_dir.endswith('.ZIP'):
        s3_object = s3_context.resource.Object(s3_context.bucket, product_dir)
        s3_file = S3File(s3_object)
        with ZipFile(s3_file) as zip_ds:
            files_list = zip_ds.namelist()
            matches = stringlist_regex(files_list, pattern)
            if use_gdal_adressing:
                return [f'/vsizip/vsis3/{s3_context.bucket}/{product_dir}/{p}' for p in matches]
            return matches
    # If we are still there, we are requesting unzipped s3 addresses, yet to be implemented
    raise NotImplementedError


@contextmanager
def agnostic_open(product_dir: str, internal_path: str, s3_context: Optional[S3Context] = None):
    """
    """
    if s3_context is None:
        # Posix case
        if product_dir.endswith('.zip') or product_dir.endswith('.ZIP'):
            # zipped posix
            with ZipFile(product_dir) as zip_file:
                with zip_file.open(internal_path) as f:
                    yield f
        else:
            # plain case
            with open(os.path.join(product_dir, internal_path), encoding='utf-8') as f:
                yield f
    else:
        if product_dir.endswith('.zip') or product_dir.endswith('.ZIP'):
            s3_object = s3_context.resource.Object(s3_context.bucket, product_dir)
            s3_file = S3File(s3_object)
            with ZipFile(s3_file) as zip_ds:
                with zip_ds.open(internal_path) as f:
                    yield f
        else:
            # If we are still there, we are requesting unzipped s3 addresses, yet to be implemented
            raise NotImplementedError


class S3File(io.RawIOBase):
    """
    From https://alexwlchan.net/2019/working-with-large-s3-objects/
    """
    def __init__(self, s3_object):
        self.s3_object = s3_object
        self.position = 0

    def __repr__(self):
        return "<%s s3_object=%r>" % (type(self).__name__, self.s3_object)

    @property
    def size(self):
        return self.s3_object.content_length

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence (%r, should be %d, %d, %d)" %
                             (whence, io.SEEK_SET, io.SEEK_CUR, io.SEEK_END))

        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        if size == -1:
            # Read to the end of the file
            range_header = "bytes=%d-" % self.position
            self.seek(offset=0, whence=io.SEEK_END)
        else:
            new_position = self.position + size

            # If we're going to read beyond the end of the object, return
            # the entire object.
            if new_position >= self.size:
                return self.read()

            range_header = "bytes=%d-%d" % (self.position, new_position - 1)
            self.seek(offset=size, whence=io.SEEK_CUR)

        return self.s3_object.get(Range=range_header)["Body"].read()

    def readable(self):
        return True
