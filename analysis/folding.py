#!/usr/bin/env python
# Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import pymysql

import config
import manage
import export

if len(sys.argv) == 1:
    print("Supported commands are: create, list, delete, load, tune")
    sys.exit(1)

COMMAND = sys.argv[1]

DB_CONN = pymysql.connect(host=config.HOST, port=config.PORT, user=config.USER, passwd=config.PASS, db=config.FOLDING_DB)
QUEUE = DB_CONN.cursor()

if COMMAND == "create":
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " create <table>")
        QUEUE.close()
        DB_CONN.close()
        sys.exit(1)
    try:
        manage.create_table(QUEUE, sys.argv[2])
    except pymysql.err.InternalError:
        pass
elif COMMAND == "list":
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " list")
        QUEUE.close()
        DB_CONN.close()
        sys.exit(1)
    try:
        for table in manage.get_tables(QUEUE):
            print(table[0])
    except pymysql.err.InternalError:
        pass
elif COMMAND == "delete":
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " delete <table>")
        QUEUE.close()
        DB_CONN.close()
        sys.exit(1)
    try:
        manage.delete_table(QUEUE, sys.argv[2])
    except pymysql.err.InternalError:
        pass
elif COMMAND == "load":
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " load <table> <input_file>")
        QUEUE.close()
        DB_CONN.close()
        sys.exit(1)
    INPUT_FILE = open(sys.argv[3])
    try:
        manage.load_file(QUEUE, sys.argv[2], INPUT_FILE)
    except:
        print(sys.exc_info())
elif COMMAND == "tune":
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " tune <table> <operator>")
        QUEUE.close()
        DB_CONN.close()
        sys.exit(1)
    try:
        CONFS = export.tune(QUEUE, sys.argv[2], sys.argv[3])
        manage.print_results(CONFS)
    except:
        print(sys.exc_info())
else:
    print("Unknown command.")
    print("Supported commands are: create, list, delete, load, tune")

QUEUE.close()
DB_CONN.commit()
DB_CONN.close()
sys.exit(0)

