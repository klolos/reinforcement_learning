
# file locations
TIRAMOLA_DIR = "/home/ubuntu/tiramola/"
LOG_FILENAME = TIRAMOLA_DIR + 'logs/Coordinator.log'

# model names
MDP        = "MDP"
MDP_CD     = "MDP-CD"
MDP_DT     = "MDP-DT"
Q_DT       = "Q-DT"
Q_LEARNING = "Q-learning"

# model settings
MODEL                   = "model"
PARAMETERS              = "parameters"
OPTIONAL_PARAMETERS     = "optional_parameters"
INITIAL_PARAMETERS      = "initial_parameters"
MAX_OPTIONAL_PARAMETERS = "max_optional_parameters"
ACTIONS                 = "actions"
DISCOUNT                = "discount"
INITIAL_QVALUES         = "initial_qvalues"
LEARNING_RATE           = "learning_rate"
TRAINING_WINDOW         = "training_window"
REWARD_IMPORTANCE       = "reward_importance"
QUALITY_RATE            = "quality_rate"
SPLIT_ERROR             = "split_error"
MIN_MEASUREMENTS        = "min_measurements"

# supported actions
ADD_VMS    = "add_VMs"
REMOVE_VMS = "remove_VMs"
NO_OP      = "no_op"
RESIZE_VMS = "resize_VMs"

# metrics calculated within tiramola
NUMBER_OF_VMS     = "number_of_VMs"
RAM_SIZE          = "RAM_size"
NUMBER_OF_CPUS    = "number_of_CPUs"
STORAGE_CAPACITY  = "storage_capacity"
PC_FREE_RAM       = "%_free_RAM"
PC_CACHED_RAM     = "%_cached_RAM"
PC_CPU_USAGE      = "%_CPU_usage"
PC_READ_THR       = "%_read_throughput"
IO_REQS           = "io_reqs"
TOTAL_LATENCY     = "total_latency"
NEXT_LOAD         = "next_load"

# metrics from ycsb
TOTAL_THROUGHPUT  = "total_throughput"
INCOMING_LOAD     = "incoming_load"
READ_THROUGHPUT   = "read_throughput"
READ_LATENCY      = "read_latency"
UPDATE_THROUGHPUT = "update_throughput"
UPDATE_LATENCY    = "update_latency"
PC_READ_LOAD      = "%_read_load"
YCSB_METRICS      = [TOTAL_THROUGHPUT, INCOMING_LOAD, READ_THROUGHPUT, READ_LATENCY, 
                     UPDATE_THROUGHPUT, UPDATE_LATENCY, PC_READ_LOAD]

# ganglia metrics from the cluster
BYTES_IN          = "bytes_in"
BYTES_OUT         = "bytes_out"
CPU_IDLE          = "cpu_idle"
CPU_NICE          = "cpu_nice"
CPU_SYSTEM        = "cpu_system"
CPU_USER          = "cpu_user"
CPU_WIO           = "cpu_wio"
DISK_FREE         = "disk_free"
LOAD_ONE          = "load_one"
LOAD_FIVE         = "load_five"
LOAD_FIFTEEN      = "load_fifteen"
MEM_BUFFERS       = "mem_buffers"
MEM_CACHED        = "mem_cached"
MEM_FREE          = "mem_free"
MEM_SHARED        = "mem_shared"
MEM_TOTAL         = "mem_total"
PART_MAX_USED     = "part_max_used"
PACKETS_IN        = "pkts_in"
PACKETS_OUT       = "pkts_out"
PROC_RUN          = "proc_run"
PROC_TOTAL        = "proc_total"
CLUSTER_METRICS   = [BYTES_IN, BYTES_OUT, CPU_IDLE, CPU_NICE, CPU_SYSTEM, CPU_USER, CPU_WIO, DISK_FREE,
                     LOAD_ONE, LOAD_FIVE, LOAD_FIFTEEN, MEM_BUFFERS, MEM_CACHED, MEM_FREE, MEM_SHARED,
                     MEM_TOTAL, PART_MAX_USED, PACKETS_IN, PACKETS_OUT, PROC_RUN, PROC_TOTAL]

NETWORK_USAGE     = "network_usage"

# ganglia metrics from the iaas
CPU_IAAS          = 'cpu'
NUMBER_OF_THREADS = "number_of_threads"
IO_READ_REQS      = "read_io_reqs"
IO_WRITE_REQS     = "write_io_reqs"
IAAS_METRICS      = [CPU_IAAS, NUMBER_OF_THREADS, IO_READ_REQS, IO_WRITE_REQS]


# values for parameters
VALUES = "values"
LIMITS = "limits"

# update algorithms
NO_UPDATE            = "no_update"
SINGLE_UPDATE        = "single_update"
VALUE_ITERATION      = "value_iteration"
PRIORITIZED_SWEEPING = "prioritized_sweeping"
UPDATE_ALGORITHMS    = [NO_UPDATE, SINGLE_UPDATE, VALUE_ITERATION, PRIORITIZED_SWEEPING]

# splitting algorithms for MDP-DT
MID_POINT      = "mid_point"
ANY_POINT      = "any_point"
MEDIAN_POINT   = "median_point"
MAX_POINT      = "max_point"
QVALUE_DIFF    = "q-value_difference"
SPLIT_CRITERIA = [MID_POINT, ANY_POINT, MEDIAN_POINT, MAX_POINT, QVALUE_DIFF]

# statistical tests
STUDENT_TTEST      = "student_ttest"
WELCH_TTEST        = "welch_ttest"
MANN_WHITNEY_UTEST = "mann_whitney_utest"
KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"

