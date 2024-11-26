-- Import in hbtn_0c_0 database this table dump
-- mysql -u root -p hbtn_0c_0 < temperatures.sql
-- CREATE DATABASE IF NOT EXISTS hbtn_0c_0;

SELECT city, AVG(value)
FROM temperatures
GROUP BY city
ORDER BY AVG(value) DESC;