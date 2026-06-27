-- Running the core data profiling engine
SUMMARIZE SELECT * FROM 'train.csv';
-- Step 2: Checking the distribution and volume of target classes
SELECT 
    class, 
    COUNT(*) AS total_records,
    ROUND(COUNT(*) * 100.0 / 577347, 2) AS data_percentage
FROM 'train.csv'
GROUP BY class
ORDER BY total_records DESC;
-- Step 3: Auditing negative redshift anomalies across target classes
SELECT 
    class, 
    COUNT(*) AS negative_redshift_records
FROM 'train.csv'
WHERE redshift < 0
GROUP BY class;
-- Step 4: Auditing negative anomalies in the 'u' filter band
SELECT 
    class, 
    COUNT(*) AS negative_u_records
FROM 'train.csv'
WHERE u < 0
GROUP BY class;
-- Step 5: Isolating and inspecting the single rogue outlier row in 'u' band
SELECT * FROM 'train.csv' 
WHERE u < 0;
-- Step 6: Multivariate Relational Audit - Checking the bounds of color differentials
SELECT 
    class,
    MIN(u - g) AS min_ug_diff,
    MAX(u - g) AS max_ug_diff,
    MIN(g - r) AS min_gr_diff,
    MAX(g - r) AS max_gr_diff
FROM 'train.csv'
GROUP BY class;
-- Step 7: Executing Production-Grade Data Cleaning & Transformation Pipeline
CREATE OR REPLACE TABLE clean_train AS
SELECT 
    id,
    alpha,
    delta,
    u,
    g,
    r,
    i,
    z,
    -- Handling systematic negative redshift errors by capping them at 0
    CASE WHEN redshift < 0 THEN 0 ELSE redshift END AS redshift,
    spectral_type,
    galaxy_population,
    class
FROM 'train.csv'
-- Dropping the isolated rogue rows that corrupt the color distribution
WHERE u > 0;
-- Step 8: Post-treatment Audit - Verifying the health of the clean dataset
SUMMARIZE SELECT * FROM clean_train;