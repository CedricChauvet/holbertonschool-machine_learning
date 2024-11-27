-- Write a SQL script that lists all bands with Glam rock as their main style, ranked by their longevity

SELECT t1.band_name,  IFNULL(t1.split, 2020)  - t1.formed  AS lifespan
FROM metal_bands t1
WHERE t1.style  LIKE '%Glam rock%'
ORDER BY lifespan DESC;