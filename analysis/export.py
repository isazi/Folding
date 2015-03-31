#!/usr/bin/env python
# Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
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

import manage

def tune(queue, table, operator):
    confs = list()
    if operator.casefold() == "max" or operator.casefold() == "min":
        dms_range = manage.get_dm_range(queue, table)
        for dm in dms_range:
            internal_list = list()
            period_range = manage.get_period_range(queue, table, dm[0])
            for period in period_range:
                queue.execute("SELECT DMsPerBlock,PeriodsPerBlock,BinsPerBlock,DMsPerThread,PeriodsPerThread,BinsPerThread,vector,GFLOPS,time,time_err,cov FROM " + table + " WHERE (GFLOPS = (SELECT " + operator + "(GFLOPS) FROM " + table + " WHERE (nrDMs = " + str(dm[0]) + " AND nrPeriods = " + str(period[0]) + ")) AND (nrDMs = " + str(dm[0]) + " AND nrPeriods = " + str(period[0]) + "))")
                best = queue.fetchall()
                internal_list.append([dm[0], period[0], best[0][0], best[0][1], best[0][2], best[0][3], best[0][4], best[0][5], best[0][6], best[0][7], best[0][8], best[0][9], best[0][10]])
            confs.append(internal_list)
    return confs

