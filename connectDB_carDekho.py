# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:19:04 2022

@author: pratiksanghvi
"""
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
        'secure_connect_bundle': 'D:\Learn & Projects\Data Science Extra\Datasets\CarDekho\secure-connect-cardekho.zip'
}
auth_provider = PlainTextAuthProvider('eTSdKCzhtbNFXQvwMSdmmNXu', 'qLwooG6sbbPIZdMJYT8JXfDeEtPasOqS3F-vwnHM.Optp0nIenAqWXtzDmXqIvv74+H5I-NR1ZH4Zr977iK2BQlTqnZb4-iWtP,YBmxKGhv8O0-Q+EFn-MMtF9BwtePI')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
    print(row[0])
else:
    print("An error occurred.")