import requests
 
url = "/api/v1/prediction/:id"
 
def query(payload):
  response = requests.post(
    url,
    json = payload
  )
  return response.json()
 
output = query({
  question: "hello!"
)}