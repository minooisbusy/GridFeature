#include<iostream>
#include<list>


int main(int argc, char*argv[])
{
    std::list<int> lt;
    // front returns reference of element.
    // begin returns iterator of element

    lt.push_back(10);
    lt.push_back(20);
    lt.push_back(30);
    std::list<int>::iterator it = lt.begin();
    std::cout<<"Initial Iterator : " <<*it<<std::endl;
    it++;
    std::cout<<"modified Iterator : " <<*it<<std::endl;
    it = lt.erase(it);
    std::cout<<"After erase method : " <<*it<<std::endl;

    


    std::cout<<lt.back()<<std::endl;
    std::cout<<lt.size()<<std::endl;


    return 0;
}