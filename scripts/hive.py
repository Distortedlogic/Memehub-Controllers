import json

from beem import Hive
from beem.account import Account
from beem.blockchain import Blockchain
from beem.comment import Comment
from beem.instance import set_shared_blockchain_instance
from beem.nodelist import NodeList
from beem.utils import construct_authorperm
from decouple import config

nodelist = NodeList()
nodelist.update_nodes()
nodes = nodelist.get_hive_nodes()
hive = Hive(node=nodes)
hive.wallet.wipe(True)
hive.wallet.unlock("wallet-passphrase")
hive.wallet.addPrivateKey(config("ACTIVE_WIF"))
set_shared_blockchain_instance(hive)
account = Account(config("HIVE_ACCOUNT"), blockchain_instance=hive)

if __name__ == "__main__":
    entries = account.get_blog_entries(start_entry_id=150, limit=1)
    for entry in entries:
        print(
            json.dumps(
                Comment(
                    construct_authorperm(entry["author"], entry["permlink"])
                ).json(),
                indent=4,
            )
        )
    # mana = account["voting_manabar"]["current_mana"]
    # blockchain = Blockchain()
    # for op in blockchain.stream("comment"):
    #     if not op["parent_author"]:
    #         print(json.dumps(op, indent=4, default=str))
