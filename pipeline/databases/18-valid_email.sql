-- Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.

DELIMITER //

-- creation of the trigger
DROP TRIGGER IF EXISTS modify_email;
CREATE TRIGGER modify_email

BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    -- if the email has been changed, the valid_email attribute is reset to 0
    IF NEW.email != OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END //

DELIMITER ;