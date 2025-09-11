import timeit

# 测试创建10000次parser的总时间
total_time = timeit.timeit("xml.sax.make_parser()", setup="import xml.sax", number=10000)
avg_time_us = (total_time / 10000) * 1e6  # 转换为单次平均时间（微秒）

print(f"单次make_parser()的平均时间：{avg_time_us:.2f} μs")
