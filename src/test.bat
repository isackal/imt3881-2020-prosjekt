IF EXIST "htmlcov" (
    rmdir /s /q htmlcov
)
IF EXIST ".coverage" (
    del .coverage
)

coverage run --source=. -m unittest test.py
coverage html
