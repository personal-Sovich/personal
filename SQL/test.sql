-- Connect server to python (host='localhost', user='root', passwd='password')

-- Press the + under MYSQL and enter the following creditionals to set the server/database
    -- host = localhost
    -- user = root
    -- passwd = password
    -- port = 3306

-- Good to use to check your queries directly in python rather than uising the mySQL workspace
-- Use SHIFT + Q TO RUN HIGHLIGHTED QUERY OR ICON IN TOP-RIGHT CORNER

SELECT SUM(amount)
FROM `payment` as p;

-- SELECT *
-- FROM city
-- WHERE Name LIKE 'K%'
-- LIMIT 10;

-- SELECT COUNT(*)
-- FROM `city`
-- WHERE Population > 1000000;
