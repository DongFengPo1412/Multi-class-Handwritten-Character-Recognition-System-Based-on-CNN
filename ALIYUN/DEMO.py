# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import sys
import json

from typing import List

from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> ocr_api20210707Client:
        """
        使用凭据初始化账号Client
        @return: Client
        @throws Exception
        """
        # 工程代码建议使用更安全的无AK方式，凭据配置方式请参见：https://help.aliyun.com/document_detail/378659.html。
        credential = CredentialClient()
        config = open_api_models.Config(
            credential=credential
        )
        # Endpoint 请参考 https://api.aliyun.com/product/ocr-api
        config.endpoint = f'ocr-api.cn-hangzhou.aliyuncs.com'
        return ocr_api20210707Client(config)

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        recognize_all_text_request = ocr_api_20210707_models.RecognizeAllTextRequest(
            url='https://img.alicdn.com/imgextra/i1/O1CN01vCBpJz1hGYOgIkp4l_!!6000000004250-0-tps-564-294.jpg',
            type='General'
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = client.recognize_all_text_with_options(recognize_all_text_request, runtime)
            print(json.dumps(resp, default=str, indent=2))
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        recognize_all_text_request = ocr_api_20210707_models.RecognizeAllTextRequest(
            url='https://img.alicdn.com/imgextra/i1/O1CN01vCBpJz1hGYOgIkp4l_!!6000000004250-0-tps-564-294.jpg',
            type='General'
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = await client.recognize_all_text_with_options_async(recognize_all_text_request, runtime)
            print(json.dumps(resp, default=str, indent=2))
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))


if __name__ == '__main__':
    Sample.main(sys.argv[1:])
