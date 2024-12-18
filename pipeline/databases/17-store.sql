-- Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.

DELIMITER //
DROP TRIGGER IF EXISTS decrease_quantity_after_order;
CREATE TRIGGER decrease_quantity_after_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items i
    SET i.quantity = i.quantity - NEW.number
    WHERE i.name = NEW.item_name;
END //

DELIMITER ;