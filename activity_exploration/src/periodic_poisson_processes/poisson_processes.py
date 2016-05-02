#!/usr/bin/env python


import time
import rospy
import datetime
from activity_exploration.msg import PoissonMsg
from periodic_poisson_processes.rate import Lambda
from mongodb_store.message_store import MessageStoreProxy


class PoissonProcesses(object):

    def __init__(self, time_window=10, minute_increment=1, coll="poisson_processes"):
        # time_window and minute_increment are in minutes
        if 60 % time_window != 0 and 60 % minute_increment != 0:
            rospy.logwarn("Time window and minute increment are not factors of 60")
            rospy.logwarn("Using default ones (time window = 10 minutes, increment = 1 minute)")
            time_window = 10
            minute_increment = 1
        self._init_time = None
        self.poisson = dict()
        self.time_window = rospy.Duration(time_window*60)
        self.minute_increment = rospy.Duration(minute_increment*60)
        self._db = MessageStoreProxy(collection=coll)

    def default_lambda(self):
        return Lambda()

    def update(self, start_time, count):
        # no end time, assuming that the count was taken within time window
        # starting from the init_time
        start_time = rospy.Time(start_time.secs)
        end_time = start_time + self.time_window
        if self._init_time is None:
            self._init_time = start_time
        key = "%s-%s" % (start_time.secs, end_time.secs)
        if key not in self.poisson:
            self.poisson[key] = Lambda()
        self.poisson[key].update_lambda([count])
        rospy.loginfo(
            "Poisson model is updated from %d to %d" % (start_time.secs, end_time.secs)
        )

    def retrieve(self, start_time, end_time):
        """ retrieve poisson distribution from specified start_time until specified end_time
            minus time window interval.
        """
        # convert start_time and end_time to the closest time range (in minutes)
        start_time, end_time = self._convert_time(start_time, end_time)
        rospy.loginfo(
            "Retrieving Poisson model from %d to %d" % (start_time.secs, end_time.secs)
        )
        if self._init_time is None:
            rospy.logwarn("Retrieving data is not possible, no poisson model has been learnt.")
            return
        end_time = end_time - self.time_window
        result = dict()
        while start_time <= end_time:
            mid_end = start_time + self.time_window
            key = "%s-%s" % (start_time.secs, mid_end.secs)
            try:
                result[key] = self.poisson[key].get_rate()
            except:
                result[key] = self.default_lambda().get_rate()
            start_time = start_time + self.minute_increment
        return result

    def _convert_time(self, start_time, end_time):
        new_start = datetime.datetime.fromtimestamp(start_time.secs)
        new_start = datetime.datetime(
            new_start.year, new_start.month, new_start.day, new_start.hour,
            new_start.minute
        )
        new_start = rospy.Time(time.mktime(new_start.timetuple()))
        new_end = datetime.datetime.fromtimestamp(end_time.secs)
        new_end = datetime.datetime(
            new_end.year, new_end.month, new_end.day, new_end.hour,
            new_end.minute
        )
        new_end = rospy.Time(time.mktime(new_end.timetuple()))
        return new_start, new_end

    def store_to_mongo(self, meta=dict()):
        rospy.loginfo("Storing all poisson data...")
        for key in self.poisson.iterkeys():
            start_time = rospy.Time(int(key.split("-")[0]))
            self._store(start_time, meta)

    def _store(self, start_time, meta):
        if self._init_time is None:
            rospy.logwarn("Storing data is not possible, no poisson model has been learnt.")
            return

        start = datetime.datetime.fromtimestamp(start_time.secs)
        end_time = start_time + self.time_window
        key = "%s-%s" % (start_time.secs, end_time.secs)
        lmbd = self.poisson[key]
        msg = PoissonMsg(
            start.month, start.day, start.hour, start.minute,
            self.time_window, lmbd.shape, lmbd.scale, lmbd.get_rate()
        )
        meta.update({"start": self._init_time.secs, "year": start.year})
        print "Storing %s with meta %s" % (str(msg), str(meta))
        query = {
            "month": start.month, "day": start.day, "hour": start.hour,
            "minute": start.minute, "duration.secs": self.time_window.secs
        }
        if len(self._db.query(PoissonMsg._type, query, meta)) > 0:
            self._db.update(msg, message_query=query, meta_query=meta)
        else:
            self._db.insert(msg, meta)

    def retrieve_from_mongo(self, meta=dict()):
        query = {
            "duration.secs": self.time_window.secs
        }
        logs = self._db.query(PoissonMsg._type, query, meta)
        if len(logs) > 0:
            rospy.loginfo("Clearing current poisson distributions...")
            self.poisson = dict()
            self._init_time = rospy.Time.now()
            for log in logs:
                if log[1]['start'] < self._init_time.secs:
                    self._init_time = rospy.Time(log[1]['start'])
                start = datetime.datetime(
                    log[1]['year'], log[0].month, log[0].day,
                    log[0].hour, log[0].minute
                )
                start = rospy.Time(time.mktime(start.timetuple()))
                end = start + self.time_window
                key = "%s-%s" % (start.secs, end.secs)
                self.poisson[key] = self.default_lambda()
                self.poisson[key].scale = log[0].scale
                self.poisson[key].shape = log[0].shape
                self.poisson[key].set_rate(log[0].rate)
        rospy.loginfo("%d new poisson distributions are obtained from db..." % len(logs))


class PeriodicPoissonProcesses(PoissonProcesses):

    def __init__(
        self, time_window=10, minute_increment=1, periodic_cycle=10080,
        coll="poisson_processes"
    ):
        self.periodic_cycle = periodic_cycle
        super(PeriodicPoissonProcesses, self).__init__(time_window, minute_increment, coll)

    def update(self, start_time, count):
        real_start = start_time
        end_time = start_time + self.time_window
        if self._init_time is not None:
            while (start_time - self._init_time) >= (self.minute_increment * self.periodic_cycle):
                start_time = start_time - (self.minute_increment * self.periodic_cycle)
        super(PeriodicPoissonProcesses, self).update(start_time, count)
        rospy.loginfo(
            "Poisson model is updated from %d to %d in the real time." % (real_start.secs, end_time.secs)
        )

    def store_to_mongo(self, meta):
        meta.update({'periodic_cycle': self.periodic_cycle})
        super(PeriodicPoissonProcesses, self).store_to_mongo(meta)

    def _store(self, start_time, meta):
        if 'periodic_cycle' not in meta:
            meta.update({'periodic_cycle': self.periodic_cycle})
        if self._init_time is not None:
            while (start_time - self._init_time) >= (self.minute_increment * self.periodic_cycle):
                start_time = start_time - (self.minute_increment * self.periodic_cycle)
        super(PeriodicPoissonProcesses, self)._store(start_time, meta)

    def retrieve_from_mongo(self, meta=dict()):
        meta.update({'periodic_cycle': self.periodic_cycle})
        super(PeriodicPoissonProcesses, self).retrieve_from_mongo(meta)

    def retrieve(self, start_time, end_time):
        # convert start_time and end_time to the closest time range (in minutes)
        start_time, end_time = super(PeriodicPoissonProcesses, self)._convert_time(
            start_time, end_time
        )
        rospy.loginfo(
            "Retrieving Poisson model from %d to %d in the real time." % (
                start_time.secs, end_time.secs
            )
        )
        result = dict()
        if self._init_time is not None:
            inter_result = list()
            real_start = start_time
            real_end = end_time
            interval = (
                self.minute_increment * (self.periodic_cycle-1)
            ) + self.time_window
            while (end_time - start_time) >= interval:
                delta_start = start_time - self._init_time
                end_time = start_time + interval
                inter_result.append(
                    super(PeriodicPoissonProcesses, self).retrieve(
                        (start_time - delta_start), (end_time - delta_start)
                    )
                )
                start_time = start_time + (
                    self.minute_increment * self.periodic_cycle
                )
                end_time = real_end
            if (end_time - start_time) >= self.time_window:
                delta_start = start_time - self._init_time
                inter_result.append(
                    super(PeriodicPoissonProcesses, self).retrieve(
                        (start_time - delta_start), (end_time - delta_start)
                    )
                )
            keys = list()
            while real_start + self.time_window <= real_end:
                mid_end = real_start + self.time_window
                key = "%s-%s" % (real_start.secs, mid_end.secs)
                keys.append(key)
                real_start = real_start + self.minute_increment
            values = list()
            for i in inter_result:
                for val in i.values():
                    values.extend([val])
            result = {key: values[ind] for ind, key in enumerate(keys)}
        return result
