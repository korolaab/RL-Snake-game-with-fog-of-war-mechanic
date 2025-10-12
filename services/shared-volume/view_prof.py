import pstats
p = pstats.Stats('./inference_profile.prof')
p.sort_stats('cumulative').print_stats(30)
#p.sort_stats('time').print_stats(100)
