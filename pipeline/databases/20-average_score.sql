-- Write a SQL script that creates a stored procedure ComputeAverageScoreForUser
-- that computes and store the average score for a student.

DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser //

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN user_id INT
)
BEGIN
    DECLARE avg_score DECIMAL(10,2);  -- Déclaration de la variable pour la moyenne

    -- Calculer la moyenne des scores pour l'utilisateur spécifié
    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE user_id = user_id;
    
    UPDATE users
    SET average_score = avg_score
    WHERE id = user_id;
END //

DELIMITER ;