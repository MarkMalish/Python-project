import csv
import simpy
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, chi2_contingency, chisquare
from scipy.stats import kstest, expon
from scipy import stats

class Params:
    def __init__(self, i1, i2, p1, p2, sd1, sd2, q):
        self.i1 = i1  # Inter-arrival time for job type 1
        self.i2 = i2  # Inter-arrival time for job type 2
        self.p1 = p1  # Processing time mean for job type 1
        self.p2 = p2  # Processing time mean for job type 2
        self.sd1 = sd1 # Standard deviation for job type 1
        self.sd2 = sd2 # Standard deviation for job type 2
        self.q = q    # Queue discipline

Job = namedtuple('Job', ['job_type', 'arrival_time'])

def get_processing_time(job, params):
    if job.job_type == 1:
        return random.normalvariate(params.p1, params.sd1)
    else:
        return random.normalvariate(params.p2, params.sd2)
    
class Server:
    def __init__(self):
        self.queue = deque()
        self.status = 0  # 0: free, 1: busy
        self.next_free_time = float('inf')  # Time when server will be free

def job_generator(env, job_type, rate, server, log, next_arrivals, params):
    while True:
        yield env.timeout(random.expovariate(1 / rate))
        arrival_time = env.now
        job = Job(job_type, arrival_time)
        next_arrivals[job_type - 1] = arrival_time + random.expovariate(1 / rate)  # Predict next arrival
        log_event(env, job, f'J{job_type}', server, log, next_arrivals, params)
        if server.status == 0 and not server.queue:
            yield env.process(process_job(env, server, job, log, next_arrivals, params))
        else:
            server.queue.append(job)
            arrival_time = random.expovariate(1 / rate)
            if server.status == 0:
                next_job = server.queue.popleft()
                yield env.process(process_job(env, server, next_job, log, next_arrivals, params))


def process_job(env, server, job, log, next_arrivals, params):
    server.status = 1
    processing_time = get_processing_time(job, params)
    server.next_free_time = env.now + processing_time
    yield env.timeout(processing_time)  # Use the processing time directly
    log_event(env, job, 'E', server, log, next_arrivals, params)
    server.status = 0
    server.next_free_time = float('inf')
    if server.queue:
        next_job = server.queue.popleft()
        env.process(process_job(env, server, next_job, log, next_arrivals, params))  # Schedule next job processing

def get_processing_time(job, params):
    # Assuming `params` contains `p1` and `p2` as processing time parameters
    return random.normalvariate(params.p1 if job.job_type == 1 else params.p2, 0.3 if job.job_type == 1 else 0.5)

def log_event(env, job, event_type, server, log, next_arrivals, params, end_time=None):
    log.append({
        'Event': event_type,
        'Tm': env.now,
        'J1': next_arrivals[0],
        'J2': next_arrivals[1],
        'St': server.next_free_time if server.status == 1 else float('inf'),
        'S': server.status,
        'n': len(server.queue),
        'Q': ', '.join(f'J{item.job_type}' for item in server.queue)
    })

def collect_simulation_data(num_runs):
    results = {'total_jobs': [], 'mean_queue_length': []}
    
    # Collect data for each simulation run
    for _ in range(num_runs):
        total_jobs, mean_queue_length = main()
        results['total_jobs'].append(total_jobs)
        results['mean_queue_length'].append(mean_queue_length)

    # Calculate statistics after all runs are completed
    results['average_total_jobs'] = np.mean(results['total_jobs'])
    results['median_total_jobs'] = np.median(results['total_jobs'])
    results['std_dev_total_jobs'] = np.std(results['total_jobs'])

    results['average_queue_length'] = np.mean(results['mean_queue_length'])
    results['median_queue_length'] = np.median(results['mean_queue_length'])
    results['std_dev_queue_length'] = np.std(results['mean_queue_length'])

    return results

# Run the simulation 10 times
def safe_chi_square(counts, expected):
    # Ensure no expected counts are zero or negative and all are at least 5
    if any(x < 5 for x in expected):
        print("Some expected counts are less than 5, which could invalidate the test results.")
        return None
    try:
        chi2_stat, p = chisquare(counts, f_exp=expected)
        return chi2_stat, p
    except ValueError as e:
        print(f"Error in Chi-square test: {e}")
        return None
def plot_exponential_data(data, rate):
    plt.figure(figsize=(12, 6))
    sns.histplot(data, bins=30, kde=False, color="blue", stat="density")
    plt.title('Histogram of Generated Exponential Data')
    x = np.linspace(0, np.max(data), 100)
    pdf = expon.pdf(x, scale=1/rate)
    plt.plot(x, pdf, 'r-', lw=2, label=f'Theoretical PDF (λ={rate})')
    plt.legend()
    plt.show()


def main():
    global params
    params = Params(1.5, 4, 2, 2.5, 0.3, 0.5, 'FIFO')
    env = simpy.Environment()
    server = Server()
    log = []
    next_arrivals = [float('inf'), float('inf')]
    env.process(job_generator(env, 1, params.i1, server, log, next_arrivals, params))
    env.process(job_generator(env, 2, params.i2, server, log, next_arrivals, params))

    # Start simulation
    log_event(env, Job(0, 0), 'start', server, log, next_arrivals, params)  # updated call
    env.run(until=501)
    # End simulation
    log_event(env, Job(0, 0), 'end', server, log, next_arrivals, params)  # updated call
    
    # Writing to CSV
    with open('simulation_log.csv', mode='w', newline='') as file:
        fieldnames = ['Event', 'Tm', 'J1', 'J2', 'St', 'S', 'n', 'Q']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in log:
            writer.writerow(event)

    total_jobs = len([event for event in log if event['Event'].startswith('J')])
    mean_queue_length = np.mean([event['n'] for event in log])

    return total_jobs, mean_queue_length

sns.set(style="whitegrid")
# Generating data from an exponential distribution
rate = 1.5  # This is the lambda (rate parameter) of the exponential distribution
data_exponential = np.random.exponential(scale=1/rate, size=1000)  # Using scale as 1/lambda



# Generate data from two different distributions
np.random.seed(123)  # For reproducibility
data_rng1 = np.random.normal(loc=0, scale=1, size=1000)  # RNG 1: Normal distribution N(0,1)
data_rng2 = np.random.uniform(low=0, high=1, size=1000)  # RNG 2: Uniform distribution U(0,1)

# Plot for RNG 1: Normal Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data_rng1, bins=30, kde=False, color="gray", stat="density")
x_values = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x_values, norm.pdf(x_values), 'r-', lw=2)  # Overlay a normal distribution curve
plt.title('Histogram of RNG 1 with Normal Fit')
plt.show()

# Plot for RNG 2: Uniform Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data_rng2, bins=30, kde=False, color="gray", stat="density")
x_values = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
plt.plot(x_values, uniform.pdf(x_values), 'b-', lw=2)  # Overlay a uniform distribution curve
plt.title('Histogram of RNG 2 with Uniform Fit')
plt.show()


# Plot histogram of the exponential data
plt.figure(figsize=(12, 6))
sns.histplot(data_exponential, bins=30, kde=False, color="blue", stat="density")
plt.title('Histogram of Generated Exponential Data')

# Overlay the theoretical PDF
x = np.linspace(0, np.max(data_exponential), 100)
pdf = expon.pdf(x, scale=1/rate)  # Exponential PDF
plt.plot(x, pdf, 'r-', lw=2, label=f'Theoretical PDF (λ={rate})')
plt.legend()

plt.show()


# Define and adjust bins for RNG 1 (Normal Distribution)
bins_rng1 = np.linspace(-3, 3, 15)  # Adjusted to more appropriate range and bin count
count_rng1, _ = np.histogram(data_rng1, bins=bins_rng1)
expected_rng1 = [1000 * (norm.cdf(bins_rng1[i+1], loc=0, scale=1) - norm.cdf(bins_rng1[i], loc=0, scale=1)) for i in range(len(bins_rng1)-1)]

# Normalize expected counts for RNG 1
expected_rng1 = np.array(expected_rng1)
expected_rng1 *= count_rng1.sum() / expected_rng1.sum()

# Define and adjust bins for RNG 2 (Uniform Distribution)
bins_rng2 = np.linspace(0, 1, 15)  # Adjusted bin count
count_rng2, _ = np.histogram(data_rng2, bins=bins_rng2)
expected_rng2 = [1000 / len(bins_rng2-1)] * (len(bins_rng2) - 1)  # Adjusted expected counts evenly

# Normalize expected counts for RNG 2
expected_rng2 = np.array(expected_rng2)
expected_rng2 *= count_rng2.sum() / expected_rng2.sum()

data = np.random.normal(loc=0, scale=1, size=1000)
bins = np.linspace(data.min(), data.max(), num=10)  # Fewer bins
hist, bin_edges = np.histogram(data, bins=bins)
# Calculate expected frequencies for a normal distribution
expected = [1000 * (norm.cdf(bin_edges[i + 1], loc=0, scale=1) - norm.cdf(bin_edges[i], loc=0, scale=1)) for i in range(len(bin_edges) - 1)]

# Normalize expected frequencies to match the sum of observed frequencies
scaling_factor = sum(hist) / sum(expected)
expected = [e * scaling_factor for e in expected]

# Combine bins if necessary to ensure each has an expected count of at least 5
while any(e < 5 for e in expected):
    for i in range(len(expected) - 1):
        if expected[i] < 5:
            expected[i] += expected.pop(i + 1)
            hist[i] += hist.pop(i + 1)
            break

# Run Chi-square test
chi2_stat, p_value = chisquare(hist, f_exp=expected)

print(f"Chi2 Statistic: {chi2_stat}, P-value: {p_value}")



# Empirical vs Theoretical parameters comparison
print(f"RNG 1 Mean: {data_rng1.mean()} (Expected: 0)")
print(f"RNG 1 SD: {data_rng1.std()} (Expected: 1)")

expected_mean2 = 0.5
expected_sd2 = np.sqrt(1/12)
print(f"RNG 2 Mean: {data_rng2.mean()} (Expected: {expected_mean2})")
print(f"RNG 2 SD: {data_rng2.std()} (Expected: {expected_sd2})")

# Kolmogorov-Smirnov Test
ks_test_rng1 = kstest(data_rng1, 'norm', args=(0, 1))
ks_test_rng2 = kstest(data_rng2, 'uniform', args=(0, 1))

print(f"K-S Test for RNG 1: p-value = {ks_test_rng1.pvalue}")
print(f"K-S Test for RNG 2: p-value = {ks_test_rng2.pvalue}")
simulation_results = collect_simulation_data(10)
print(simulation_results)
data = np.array([500, 501, 550, 600, 543, 516, 459, 524, 418, 568])

# Calculate mean, median, mode, standard deviation, and 95% confidence interval
mean = np.mean(data)
median = np.median(data)
mode_result = stats.mode(data)
mode = mode_result.mode
count = mode_result.count

# Check and handle if mode or count are not array-like
if np.isscalar(mode):
    mode = [mode]
if np.isscalar(count):
    count = [count]

# Use the first element safely
mode = mode[0]
count = count[0]

std_dev = np.std(data)
conf_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(len(data)))

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode, "with a count of", count)
print("Standard Deviation:", std_dev)
print("95% Confidence Interval:", conf_interval)

plot_exponential_data(data_exponential, rate=4)

if __name__ == '__main__':
    main()
    