string soma                 # soma map name
string soma_config          # soma configuration
string region_id            # region id from soma database
int32 day                   # start day of the interval
int32 hour                  # start hour of the interval
int32 minute                # start minute of the interval
duration duration           # time interval duration for the poisson
float32 occurrence_shape    # alpha of gamma distribution 
float32 occurrence_scale    # beta  of gamma distribution
float32 occurrence_rate     # point in gamma distribution for the poisson rate
string periodic_type        # init periodic type: weekly or monthly
