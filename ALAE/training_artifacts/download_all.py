import download
from packaging import version

try:
    download.from_google_drive('170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz', directory='training_artifacts/ffhq')
    download.from_google_drive('1QESywJW8N-g3n0Csy0clztuJV99g8pRm', directory='training_artifacts/ffhq')
    download.from_google_drive('18BzFYKS3icFd1DQKKTeje7CKbEKXPVug', directory='training_artifacts/ffhq')
except IOError:
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth', directory='training_artifacts/ffhq')
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_194.pth', directory='training_artifacts/ffhq')
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth', directory='training_artifacts/ffhq')

try:
    download.from_google_drive('1T4gkE7-COHpX38qPwjMYO-xU-SrY_aT4', directory='training_artifacts/celeba')
except IOError:
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba/model_final.pth', directory='training_artifacts/celeba')

try:
    download.from_google_drive('1gmYbc6Z8qJHJwICYDsB4aBMxXjnKeXA_', directory='training_artifacts/bedroom')
except IOError:
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/bedroom/model_final.pth', directory='training_artifacts/bedroom')

try:
    download.from_google_drive('1ihJvp8iJWcLxTIjkV5cyA7l9TrxlUPkG', directory='training_artifacts/celeba-hq256')
    download.from_google_drive('1gFQsGCNKo-frzKmA3aCvx07ShRymRIKZ', directory='training_artifacts/celeba-hq256')
except IOError:
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_262r.pth', directory='training_artifacts/celeba-hq256')
    download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_580r.pth', directory='training_artifacts/celeba-hq256')
