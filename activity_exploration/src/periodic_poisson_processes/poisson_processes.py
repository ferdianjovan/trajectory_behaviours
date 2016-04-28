#!/usr/bin/env python


import rospy
import datetime
from activity_exploration.msg import PoissonMsg
from periodic_poisson_processes.rate import Lambda
from mongodb_store.message_store import MessageStoreProxy


class PoissonProcesses(object):

    # Capped to one year data
    def __init__(self, time_window=10, minute_increment=1, coll="poisson_processes"):
        # time_window and minute_increment are in minutes
        if 60 % time_window != 0 and 60 % minute_increment != 0:
            rospy.logwarn("Time window and minute increment are not factors of 60")
            rospy.logwarn("Using default ones (time window = 10 minutes, increment = 1 minute)")
            time_window = 10
            minute_increment = 1
        self.poisson = dict()
        self.time_window = time_window
        self.minute_increment = minute_increment
        self._db = MessageStoreProxy(collection=coll)

    def default_lambda(self):
        return Lambda()

    def update(self, init_time, count):
        # no end time, assuming that the count was taken within time window
        # starting from the init_time
        start = datetime.datetime.fromtimestamp(init_time.secs)
        end = start + datetime.timedelta(minutes=self.time_window)
        if start.month not in self.poisson:
            self.poisson[start.month] = dict()
        if start.day not in self.poisson[start.month]:
            self.poisson[start.month][start.day] = dict()
        if start.hour not in self.poisson[start.month][start.day]:
            self.poisson[start.month][start.day][start.hour] = dict()
        key = "%s-%s" % (start.minute, end.minute)
        if key not in self.poisson[start.month][start.day][start.hour]:
            self.poisson[start.month][start.day][start.hour][key] = Lambda()
        self.poisson[start.month][start.day][start.hour][key].update_lambda(
            [count]
        )
        rospy.loginfo("Poisson model is updated from %s to %s" % (str(start), str(end)))

    def retrieve(self, start_time, end_time):
        """ retrieve poisson distribution from specified start_time until specified end_time
            minus time window interval.
        """
        result = dict()
        start = datetime.datetime.fromtimestamp(start_time.secs)
        end = datetime.datetime.fromtimestamp(end_time.secs)
        rospy.loginfo("Retrieving Poisson model from %s to %s" % (str(start), str(end)))
        end = end - datetime.timedelta(minutes=self.time_window)
        while start <= end:
            mid_end = start + datetime.timedelta(minutes=self.time_window)
            key = "%s-%s" % (start.minute, mid_end.minute)
            if start.month not in result:
                result[start.month] = dict()
            if start.day not in result[start.month]:
                result[start.month][start.day] = dict()
            if start.hour not in result[start.month][start.day]:
                result[start.month][start.day][start.hour] = dict()
            try:
                result[start.month][start.day][start.hour][key] = self.poisson[start.month][start.day][start.hour][key]
            except:
                result[start.month][start.day][start.hour][key] = self.default_lambda()
            start = start + datetime.timedelta(minutes=self.minute_increment)
        return result

    def store_to_mongo(self, meta=dict()):
        rospy.loginfo("Storing all poisson data...")
        for month, daily_poisson in self.poisson.iteritems():
            for day, hourly_poisson in daily_poisson.iteritems():
                for hour, minutely_poisson in hourly_poisson.iteritems():
                    for key, lmbd in minutely_poisson.iteritems():
                        minute, _ = key.split("-")
                        self._store(month, day, hour, int(minute), lmbd, meta)

    def _store(self, month, day, hour, minute, lmbd, meta):
        msg = PoissonMsg(
            month, day, hour, minute,
            rospy.Duration(self.time_window*60),
            lmbd.shape, lmbd.scale, lmbd.get_rate()
        )
        print "Storing %s with meta %s" % (str(msg), str(meta))
        query = {
            "month": month, "day": day, "hour": hour,
            "minute": minute, "duration.secs": (self.time_window*60)
        }
        if len(self._db.query(PoissonMsg._type, query, meta)) > 0:
            self._db.update(msg, message_query=query, meta_query=meta)
        else:
            self._db.insert(msg, meta)

    def retrieve_from_mongo(self, meta=dict()):
        query = {
            "duration.secs": self.time_window*60
        }
        logs = self._db.query(PoissonMsg._type, query, meta)
        logs = [log[0] for log in logs]
        rospy.loginfo("Clearing current poisson data...")
        self.poisson = dict()
        for log in logs:
            if log.month not in self.poisson:
                self.poisson[log.month] = dict()
            if log.day not in self.poisson[log.month]:
                self.poisson[log.month][log.day] = dict()
            if log.hour not in self.poisson[log.month][log.day]:
                self.poisson[log.month][log.day][log.hour] = dict()
            end = (log.minute + self.time_window) % 60
            key = "%s-%s" % (log.minute, end)
            self.poisson[log.month][log.day][log.hour][key] = self.default_lambda()
            self.poisson[log.month][log.day][log.hour][key].scale = log.scale
            self.poisson[log.month][log.day][log.hour][key].shape = log.shape
            self.poisson[log.month][log.day][log.hour][key].set_rate(log.rate)
        rospy.loginfo("New poisson data are obtained from db...")
