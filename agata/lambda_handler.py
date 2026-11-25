from mangum import Mangum

from agata.api import app

handler = Mangum(app, lifespan="on")
