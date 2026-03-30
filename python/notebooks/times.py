import numpy
from pipe import (
    Pipe
)
QUEEN_EVENTS = {
    3: [
        ('2026-03-12', 'removed'),
    ],
    4: [
        ('2026-03-09', 'removed'),
        ('2026-03-12', 'introduced'),
    ],
}
DAY_SLICES = {
    'full day':          (0, 24),
    'night only':        (0, 6),
    'broad daylight':    (8, 18),
    'foraging window':   (10, 16),
    'morning priming':   (6, 10),
    'afternoon':         (12, 18),
}

TIME_SLICES = [
    (0, 1, "00-01"),
    (1, 2, "01-02"),
    (2, 3, "02-03"),
    (3, 4, "03-04"),
    (4, 5, "04-05"),
    (5, 6, "05-06"),
    (6, 7, "06-07"),
    (7, 8, "07-08"),
    (8, 9, "08-09"),
    (9, 10, "09-10"),
    (10, 11, "10-11"),
    (11, 12, "11-12"),
    (12, 13, "12-13"),
    (13, 14, "13-14"),
    (14, 15, "14-15"),
    (15, 16, "15-16"),
    (16, 17, "16-17"),
    (17, 18, "17-18"),
    (18, 19, "18-19"),
    (19, 20, "19-20"),
    (20, 21, "20-21"),
    (21, 22, "21-22"),
    (22, 23, "22-23"),
    (23, 24, "23-00"),
]



def slice_label_from_hour(hour):
    for start_hour, end_hour, label in TIME_SLICES:
        if start_hour <= hour < end_hour:
            return label
    assert False, f"hour {hour} not in any time slice"

@Pipe
def put_time_slice_tag(stream):
    for timestamp, features, tags in stream:
        hour = int(
            (timestamp.astype("datetime64[m]") - timestamp.astype("datetime64[D]"))
            .astype(int) // 60
        )
        yield timestamp, features, {
            **tags,
            "time_slice": slice_label_from_hour(hour),
        }


queen_event_window = ("2026-03-09", "2026-03-14")
before_queen_event_window = ("2026-03-01", "2026-03-09")
after_queen_event_window = ("2026-03-14", "2026-03-30")
queen_event_month_window = ("2026-03-01", "2026-03-30")
queen_event_day_window = ("2026-03-12", "2026-03-13")

daytime = ("07:00", "20:00")
nighttime = ("20:00", "07:00")

def timestamp_filter_from_hour_range(start_time, end_time):
    start_h, start_m = (int(x) for x in start_time.split(":"))
    end_h, end_m = (int(x) for x in end_time.split(":"))
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    def is_inside(timestamp):
        dt = timestamp.astype("datetime64[m]") - timestamp.astype("datetime64[D]")
        minutes = dt.astype(int)
        if start_minutes <= end_minutes:
            return start_minutes <= minutes < end_minutes
        return minutes >= start_minutes or minutes < end_minutes

    return is_inside

is_daytime = timestamp_filter_from_hour_range(*daytime)
is_nighttime = timestamp_filter_from_hour_range(*nighttime)

def is_between_dates(start_date, end_date):
    start = numpy.datetime64(start_date)
    end = numpy.datetime64(end_date)
    return lambda timestamp: start <= timestamp < end

is_queenlessness_outer_window = is_between_dates(*queen_event_window)
is_queenlessness_outer_extended_window = is_between_dates(*queen_event_month_window)
