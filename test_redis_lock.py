# -*- coding: utf-8 -*-

import redis
import redis_lock

r0 = redis.Redis()
r1 = redis.Redis(db=1)
redis_lock.reset_all(r0)
redis_lock.reset_all(r1)

global_lock_00 = redis_lock.Lock(r0, "global_lock")
global_lock_01 = redis_lock.Lock(r0, "global_lock")

print(global_lock_00.acquire(blocking=False))  # True
print(global_lock_01.acquire(blocking=False))  # False

global_lock_10 = redis_lock.Lock(r1, "global_lock")
global_lock_11 = redis_lock.Lock(r1, "global_lock")

print(global_lock_10.acquire(blocking=False))  # True
print(global_lock_11.acquire(blocking=False))  # False

redis_lock.reset_all(r0)
redis_lock.reset_all(r1)
