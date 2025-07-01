from data_sources.hyblock_fetcher import get_liquidation_levels

def test_hyblock():
    result = get_liquidation_levels()
    if result:
        print(f"✅ Total liquidation levels: {len(result)}")
    else:
        print("❌ Gagal ambil data dari Hyblock")

if __name__ == "__main__":
    test_hyblock()

