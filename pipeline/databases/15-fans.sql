-- Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans

SELECT t1.origin AS origin, SUM(t1.fans) AS nb_fans
FROM metal_bands t1
GROUP BY t1.origin
ORDER BY nb_fans DESC;