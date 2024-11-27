-- Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.

SELECT t1.name AS genre, COUNT(t2.show_id) AS number_of_shows
FROM tv_genres t1
LEFT JOIN tv_show_genres t2 ON t1.id = t2.genre_id
GROUP BY t1.name
ORDER BY number_of_shows DESC;